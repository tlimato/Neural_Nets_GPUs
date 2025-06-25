# Author      : Tyson Limato
# Date        : 2025-6-24
# File Name   : dynamic_work_assignment.py

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def compute(x):
    return x * x

if rank == 0:
    data = list(range(20))
    num_workers = size - 1
    index = 0

    # Send initial data to workers
    for i in range(1, size):
        if index < len(data):
            comm.send(data[index], dest=i)
            index += 1

    # Receive results and send more work
    results = []
    while index < len(data):
        result = comm.recv(source=MPI.ANY_SOURCE)
        results.append(result)
        comm.send(data[index], dest=result[0])
        index += 1

    # Final receive and shutdown
    for _ in range(1, size):
        result = comm.recv(source=MPI.ANY_SOURCE)
        results.append(result)
        comm.send(None, dest=result[0])  # Stop signal

    print("Results:", [r[1] for r in sorted(results, key=lambda x: x[0])])

else:
    while True:
        task = comm.recv(source=0)
        if task is None:
            break
        result = compute(task)
        comm.send((rank, result), dest=0)
