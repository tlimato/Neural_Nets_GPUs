# Author      : Tyson Limato
# Date        : 2025-6-25
# File Name   : gpu_histogram.py

import cupy as cp
import matplotlib.pyplot as plt

# This simply looks cool
data = cp.random.randn(1_000_000)
hist, bins = cp.histogram(data, bins=100)

plt.bar(cp.asnumpy(bins[:-1]), cp.asnumpy(hist), width=0.1)
plt.title("Histogram from GPU")

plt.savefig("GPU_Histogram.png")
