# Author      : Tyson Limato
# Date        : 2025-6-18
# File Name   : cond_statements.py
from mpi4py import MPI

comm = MPI.COMM_WORLD

if comm.rank == 1:
    print ('doing the task of rank 1')
if comm.rank == 2:
    print ('doing the task of rank 2')