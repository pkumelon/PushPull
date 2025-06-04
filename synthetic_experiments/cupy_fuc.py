import cupy as cp
import numpy as np
import pandas as pd

def grad_with_batch_batched_gpu(
    x_batched,  # Shape: (num_runs, n, d)
    y_nodes_gpu,  # Shape: (n, L) - global y_tilde on GPU
    h_nodes_gpu,  # Shape: (n, L, d) - global h_tilde on GPU
    rho,
    batch_size,
    num_runs #  num_runs
):
    n_nodes, L_samples, d_dims = h_nodes_gpu.shape # n, L, d from h_tilde

    if batch_size is None or batch_size >= L_samples:
        batch_size_eff = L_samples
        # Expand h_nodes_gpu and y_nodes_gpu for num_runs using broadcasting
        # h_batch_gpu shape: (num_runs, n, L, d)
        # y_batch_gpu shape: (num_runs, n, L)
        h_batch_gpu = cp.broadcast_to(h_nodes_gpu[cp.newaxis, ...], (num_runs, n_nodes, L_samples, d_dims))
        y_batch_gpu = cp.broadcast_to(y_nodes_gpu[cp.newaxis, ...], (num_runs, n_nodes, L_samples))
    else:
        batch_size_eff = batch_size
        # Sample indices for each run and each node independently
        # batch_indices shape: (num_runs, n, batch_size_eff)
        # CuPy's random.choice doesn't directly support this multi-axis independent sampling easily.
        # We can generate for one run-node and then tile, or loop (less efficient but clear).
        # A more efficient way is to generate a large pool of random numbers.
        # For simplicity here, let's assume we can get appropriately shaped indices.
        # A practical way for (num_runs, n, batch_size_eff):
        all_indices = cp.random.rand(num_runs, n_nodes, L_samples).argsort(axis=-1)[:, :, :batch_size_eff]
        batch_indices = all_indices.astype(cp.int32) # Ensure integer indices

        # Gather h_batch and y_batch using these indices
        # h_batch_gpu shape: (num_runs, n, batch_size_eff, d)
        # y_batch_gpu shape: (num_runs, n, batch_size_eff)
        
        # Create indices for gathering
        run_idx = cp.arange(num_runs)[:, cp.newaxis, cp.newaxis] # (num_runs, 1, 1)
        node_idx = cp.arange(n_nodes)[cp.newaxis, :, cp.newaxis]   # (1, n, 1)
        
        h_batch_gpu = h_nodes_gpu[node_idx, batch_indices, :] 
        y_batch_gpu = y_nodes_gpu[node_idx, batch_indices]

    # x_batched is (num_runs, n, d)
    # h_batch_gpu is (num_runs, n, batch_size_eff, d)
    # y_batch_gpu is (num_runs, n, batch_size_eff)

    # einsum for h_dot_x: result shape (num_runs, n, batch_size_eff)
    h_dot_x = cp.einsum('rnbd,rnd->rnb', h_batch_gpu, x_batched)
    
    exp_val = cp.exp(y_batch_gpu * h_dot_x)
    cp.clip(exp_val, a_min=None, a_max=1e300, out=exp_val)

    # einsum for g1: result shape (num_runs, n, d)
    g1 = -cp.einsum('rnbd,rnb->rnd', h_batch_gpu, y_batch_gpu / (1 + exp_val)) / batch_size_eff
    
    x_squared = x_batched**2
    g2 = 2 * x_batched / (1 + x_squared)**2
    
    grad_val = (g1 + rho * g2) # Shape (num_runs, n, d)
    return grad_val # No reshape needed if einsum is correct

def PushPull_with_batch_batched_gpu(
    A_gpu, B_gpu, init_x_gpu_batched, # init_x shape: (num_runs, n, d)
    h_data_nodes_gpu, y_data_nodes_gpu, # Original h_tilde, y_tilde on GPU
    grad_func_batched_gpu, # The new batched gradient function for GPU
    rho, lr, sigma_n,
    max_it, batch_size, num_runs
):
    x = cp.copy(init_x_gpu_batched) # Shape: (num_runs, n, d)
    num_n, num_d = x.shape[1], x.shape[2] # n, d

    # Initial gradient calculation
    g = grad_func_batched_gpu(x, y_data_nodes_gpu, h_data_nodes_gpu, rho, batch_size, num_runs)
    if sigma_n > 0:
        g += sigma_n * cp.random.normal(size=(num_runs, num_n, num_d))
    
    y = cp.copy(g) # Shape: (num_runs, n, d)

    # Store average gradient norm over runs
    avg_gradient_norm_history = []

    for iter_num in range(max_it):
        # x_update: x = A @ x - lr * y
        # A_gpu is (n,n). x is (num_runs, n, d).
        # einsum: 'jk,rkl->rjl' where j=n_out, k=n_in, r=num_runs, l=d
        term_Ax = cp.einsum('jk,rkl->rjl', A_gpu, x)
        x = term_Ax - lr * y
        
        # New gradient
        g_new = grad_func_batched_gpu(x, y_data_nodes_gpu, h_data_nodes_gpu, rho, batch_size, num_runs)
        if sigma_n > 0:
            g_new += sigma_n * cp.random.normal(size=(num_runs, num_n, num_d))

        # y_update: y = B @ y + g_new - g
        term_By = cp.einsum('jk,rkl->rjl', B_gpu, y)
        y = term_By + g_new - g
        g = g_new # Update old gradient

        # --- Record history (averaged over runs) ---
        # 1. Calculate mean_x for each run: x_mean_per_run shape (num_runs, 1, d)
        x_mean_per_run = cp.mean(x, axis=1, keepdims=True)
        
        # 2. Expand x_mean_per_run for grad_func: shape (num_runs, n, d)
        x_mean_expand_per_run = cp.broadcast_to(x_mean_per_run, (num_runs, num_n, num_d))
        
        # 3. Calculate full batch gradient for each run's x_mean: _grad_on_full_per_run shape (num_runs, n, d)
        #    Use batch_size=None for full dataset, no sigma_n noise for this evaluation
        _grad_on_full_per_run = grad_func_batched_gpu(
            x_mean_expand_per_run, y_data_nodes_gpu, h_data_nodes_gpu,
            rho=rho, batch_size=None, num_runs=num_runs
        )
        
        # 4. Calculate mean gradient over nodes for each run: mean_grad_per_run shape (num_runs, 1, d)
        mean_grad_per_run = cp.mean(_grad_on_full_per_run, axis=1, keepdims=True)
        
        # 5. Calculate norm of mean_grad for each run: norm_per_run shape (num_runs,)
        norm_per_run = cp.linalg.norm(mean_grad_per_run, axis=2).squeeze() # Squeeze out d-dim (norm result) and then 1-dim
        
        # 6. Average these norms over all runs: avg_norm_over_runs (scalar)
        avg_norm_over_runs_scalar = cp.mean(norm_per_run)
        avg_gradient_norm_history.append(cp.asnumpy(avg_norm_over_runs_scalar)) # Store as numpy float for pandas

        if (iter_num + 1) % 10 == 0: # Print progress
            print(f"Iteration {iter_num+1}/{max_it}, Avg Grad Norm: {avg_norm_over_runs_scalar:.6f}")


    return pd.DataFrame({
        "gradient_norm_on_full_trainset_avg": avg_gradient_norm_history,
    })