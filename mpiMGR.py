# Author      : Tyson Limato
# Date        : 2025-6-18
# File Name   : mpiMGR.py
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
        # Initialize the MPI communicator
        self.comm = MPI.COMM_WORLD
        # Get the rank (ID) of the current process
        self.rank = self.comm.Get_rank()
        # Get the total number of processes
        self.size = self.comm.Get_size()

    def broadcast_model(self, model, root: int = 0):
        """
        Broadcast the weights and biases of each trainable layer in the model
        from the root process to all other MPI processes.

        Parameters:
        -----------
        model : object
            The model with a `full_layers` attribute that contains all layers.
        root : int
            The rank of the process to broadcast the model from (default is 0).
        """
        for layer in model.full_layers:
            if hasattr(layer, 'weights'):
                # Broadcast weights from the root process to all others
                layer.weights = self.comm.bcast(layer.weights, root=root)
                # Broadcast biases from the root process to all others
                layer.biases  = self.comm.bcast(layer.biases,  root=root)

    def average_gradients(self, grads: dict) -> dict:
        """
        Average the gradients across all MPI processes using an all-reduce sum.

        Parameters:
        -----------
        grads : dict
            A dictionary mapping layer indices to their gradient values.

        Returns:
        --------
        dict
            A dictionary with the same keys but values averaged over all processes.
        """
        averaged = {}
        for idx, grad in grads.items():
            # Sum gradients across all processes
            total = self.comm.allreduce(grad, op=MPI.SUM)
            # Average by dividing by the number of processes
            averaged[idx] = total / self.size
        return averaged
