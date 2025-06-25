# Author      : Tyson Limato
# Date        : 6/25/2025
# File Name   : mpiMGR_ASSIGNMENT.py
from mpi4py import MPI

class MPIManager:
    """
    A utility class to handle MPI operations for distributed training using `mpi4py`.

    Methods:
    --------
    broadcast_model(model, root=0)
        Broadcasts the weights and biases of all layers in a model from the root process
        to all other processes.

    average_gradients(grads)
        Averages gradients across all MPI processes using all-reduce.
    """
    def __init__(self):
        # TODO: Initialize the MPI communicator

        self.comm = None
        
        # TODO: Get the rank (ID) of the current process

        self.rank = None
        
        # TODO: Get the total number of processes
        self.size = None

    def broadcast_model(self, model, root: int = 0):
        """
        Broadcast model parameters (weights & biases) from root to all ranks.

        Parameters:
        -----------
        model : object
            The model with a `full_layers` attribute containing all layers.
        root : int
            The rank that holds the source parameters.
        """
        # TODO: Loop over each layer in model.full_layers
        #       If a layer has `weights`, broadcast weights and biases.
        pass

    def average_gradients(self, grads: dict) -> dict:
        """
        Average the gradients across all MPI ranks.

        Parameters:
        -----------
        grads : dict
            Mapping from layer index to gradient array.

        Returns:
        --------
        dict
            Gradient dictionary with averaged values.
        """
        averaged = {}
        # TODO: For each (idx, grad) in grads:
        #         1) Sum gradients over all ranks using allreduce
        #         2) Divide by self.size to get the average
        #         3) Store in averaged[idx]
        return averaged
