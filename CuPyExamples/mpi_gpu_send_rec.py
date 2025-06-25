# Author      : Tyson Limato
# Date        : 2025-6-25
# File Name   : mpi_gpu_send_rec.py

from mpi4py import MPI
import cupy as cp
import numpy as np

# Make sure to execute with 2 cores: mpiexec -n 2 python mpi_gpu_send_rec.py
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    # Create some data on the GPU
    gpu_data = cp.arange(10, dtype=cp.float32) * 2
    print(f"[Rank 0] Original GPU data:\n{gpu_data}")

    # Move it to CPU (host) memory
    cpu_data = cp.asnumpy(gpu_data)

    # Send to rank 1
    comm.send(cpu_data, dest=1, tag=42) #tag val doesn't particularly matter as long as its unique per device

elif rank == 1:
    # Receive from rank 0
    cpu_data = comm.recv(source=0, tag=42) #tag val doesn't particularly matter as long as its unique per device

    # Move it to GPU
    gpu_data = cp.asarray(cpu_data)
    print(f"[Rank 1] Received data on GPU:\n{gpu_data}")

    # Optional: do a GPU operation
    gpu_result = cp.sqrt(gpu_data)
    print(f"[Rank 1] After GPU operation (sqrt):\n{gpu_result}")
