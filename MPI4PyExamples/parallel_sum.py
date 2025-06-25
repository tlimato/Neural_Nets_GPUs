# Author      : Tyson Limato
# Date        : 2025-6-24
# File Name   : parallel_sum.py

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Example data on root
if rank == 0:
    data = list(range(100))  # Sum from 0 to 99
    chunks = [data[i::size] for i in range(size)]
else:
    chunks = None

# Scatter data to all ranks
my_chunk = comm.scatter(chunks, root=0)

# Local computation
local_sum = sum(my_chunk)

# Reduce to get total sum on rank 0
total_sum = comm.reduce(local_sum, op=MPI.SUM, root=0)

if rank == 0:
    print(f"Total sum = {total_sum}")
