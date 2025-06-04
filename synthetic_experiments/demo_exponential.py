from useful_functions_with_batch import *
from network_utils import *
import cupy as cp
from cupy_fuc import grad_with_batch_batched_gpu, PushPull_with_batch_batched_gpu

d = 10
L_total = 1440000
n = 16
num_runs = 20
device_id = "cuda:0"
rho = 1e-2
lr = 5e-2
max_it = 3000
bs = 200


gpu_id_int = int(device_id.split(":")[1])
cp.cuda.Device(gpu_id_int).use()
print(f"Using GPU: {cp.cuda.Device(gpu_id_int).pci_bus_id}")

h_global_cpu, y_global_cpu, x_opt_cpu = init_global_data(d=d, L_total=L_total, seed=42)
print("h:", h_global_cpu.shape)
print("y:", y_global_cpu.shape)

h_tilde_cpu, y_tilde_cpu = distribute_data(h=h_global_cpu, y=y_global_cpu, n=n)
print("h_tilde:", h_tilde_cpu.shape)
print("y_tilde:", y_tilde_cpu.shape)

init_x_cpu_single = init_x_func(n=n, d=d, seed=42)
A_cpu, B_cpu = generate_exp_matrices(n=n, seed=42)
print("CPU data is prepared.")


A_gpu = cp.asarray(A_cpu)
B_gpu = cp.asarray(B_cpu)
h_tilde_gpu_nodes = cp.asarray(h_tilde_cpu)
y_tilde_gpu_nodes = cp.asarray(y_tilde_cpu)


init_x_gpu_batched = cp.repeat(
    cp.asarray(init_x_cpu_single)[cp.newaxis, ...], num_runs, axis=0
)


print("Data moved to GPU.")
print("A_gpu shape:", A_gpu.shape)
print("h_tilde_gpu_nodes shape:", h_tilde_gpu_nodes.shape)
print("init_x_gpu_batched shape:", init_x_gpu_batched.shape)

print(
    f"\nStarting batched experiment with n={n}, num_runs={num_runs} on GPU {device_id}"
)

L1_avg_df = PushPull_with_batch_batched_gpu(
    A_gpu=A_gpu,
    B_gpu=B_gpu,
    init_x_gpu_batched=init_x_gpu_batched,
    h_data_nodes_gpu=h_tilde_gpu_nodes,
    y_data_nodes_gpu=y_tilde_gpu_nodes,
    grad_func_batched_gpu=grad_with_batch_batched_gpu,
    rho=rho,
    lr=lr,
    sigma_n=0, # manually setted noise, here it is 0
    max_it=max_it,
    batch_size=bs,
    num_runs=num_runs,
)
print("\nL1_avg_df (from GPU batched execution):")
print(L1_avg_df.head())


output_path = (
    f"./EXP_out/EXP_avg_n={n}_gpu_batched.csv"
)
L1_avg_df.to_csv(output_path, index_label="iteration")
print(f"Average results saved to {output_path}")


cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()
