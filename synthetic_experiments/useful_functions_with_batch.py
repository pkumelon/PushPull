import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st
import networkx as nx
from mpmath import mp

def init_data(n=6, d=5, L=200, seed=42, sigma_h=10):
    np.random.seed(seed)
    x_opt = np.random.normal(size=(1, d))
    x_star = x_opt + sigma_h * np.random.normal(size=(n, d))
    h = np.random.normal(size=(n, L, d))
    y = np.zeros((n, L))
    for i in range(n):
        for l in range(L):
            z = np.random.uniform(0, 1)
            if 1 / z > 1 + np.exp(-np.inner(h[i, l, :], x_star[i])):
                y[i, l] = 1
            else:
                y[i, l] = -1
    return (h, y, x_opt, x_star)


def init_x_func(n=6, d=10, seed=42):
    np.random.seed(seed)
    return 0.01 * np.random.normal(size=(n, d))


def init_global_data(d=5, L_total=200, seed=42):
    np.random.seed(seed)
    x_opt = np.random.normal(size=(1, d))

    h = np.random.normal(size=(L_total, d))

    y = np.zeros(L_total)
    for l in range(L_total):
        z = np.random.uniform(0, 1)
        if 1 / z > 1 + np.exp(-np.dot(h[l], x_opt.flatten())):
            y[l] = 1
        else:
            y[l] = -1

    return h, y, x_opt


def distribute_data(h, y, n):
    L_total = h.shape[0]
    assert L_total % n == 0, "L_total must be divisible by n"
    L_per_node = L_total // n

    h_tilde = h.reshape(n, L_per_node, -1)
    y_tilde = y.reshape(n, L_per_node)

    return h_tilde, y_tilde
