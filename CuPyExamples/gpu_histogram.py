import cupy as cp
import matplotlib.pyplot as plt

# This simply looks cool
data = cp.random.randn(1_000_000)
hist, bins = cp.histogram(data, bins=100)

plt.bar(cp.asnumpy(bins[:-1]), cp.asnumpy(hist), width=0.1)
plt.title("Histogram from GPU")
# Visualizes the default gaussian distribution of rand numbers
plt.savefig("GPU_Histogram.png")
