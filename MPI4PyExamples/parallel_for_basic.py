# Author      : Tyson Limato
# Date        : 2025-6-24
# File Name   : parallel_for_basic.py

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Work: Iterate over this list in parallel
work = list(range(20))

# Divide work among processes (static partitioning)
chunk = work[rank::size]

print(f"Rank {rank} handling: {chunk}")
