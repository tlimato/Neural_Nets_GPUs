from mpi4py import MPI

comm = MPI.COMM_WORLD

print('HI, my rank is:', comm.rank)

# Test Run
'''
(mpi-env) (mpi-env) (base) tylim@DESKTOP-KDPTHA0:/mnt/c/Nueral_Nets_GPUs/MPI4PyExamples$ mpirun.openmpi -np 2 python hi_rank.py 
HI, my rank is: 1
HI, my rank is: 0
'''