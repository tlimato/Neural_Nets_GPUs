# ------------------------------------------------------------
# Author      : Tyson Limato
# Date        : 2025-6-01
# File Name   : testing.py
# Description : This code demonstrates the various types of parallelism
#               by training a multilayer perceptron to predict housing
#               prices based on 14 different metrics using 9 Dense layers and
#               8 ReLu hidden layers.
#
# Usage       : Describe how to run the script and any required
#               command-line arguments or dependencies.
#
# Dependencies:
#   See the package-list.txt file for all of the utilized packages.
#   To easily replicate the environment install it using conda or miniconda3
#   using: "conda create --name <env> --file package-list.txt". The large dependencies
#   are below.
#       - mpi4py
#       - pandas
#       - numpy
#       - cupy (I used the Cuda 12x variant)
#
# Notes:
#   - Complete multi-GPU testing
#   - implement graphs
#   -
#
# ------------------------------------------------------------

from mpi4py import MPI
import pandas as pd
import numpy as np
import json
import time
import argparse
import os
import cupy as cp
# ------------------ MPI Communication Manager ------------------
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


# ------------------ Layer Base ------------------

class Layer:
    """
    Abstract base class for neural network layers.

    This class defines the interface that all custom layers must implement
    to be compatible with the rest of the model's forward and backward passes.

    Methods:
    --------
    forward(x: np.ndarray) -> np.ndarray
        Computes the output of the layer for a given input.

    backward(grad: np.ndarray)
        Computes the gradient of the loss with respect to the layer's input
        using the gradient from the next layer.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad: np.ndarray):
        raise NotImplementedError


# ------------------ Dense Layer ------------------
import numpy as np

class Dense(Layer):
    """
    A fully connected (dense) layer for a neural network.

    This layer performs a linear transformation: y = W·x + b

    Parameters:
    -----------
    in_dim : int
        Number of input features.
    out_dim : int
        Number of output neurons.
    lr : float
        Learning rate used for parameter updates (default: 0.001).

    Methods:
    --------
    forward(x: np.ndarray) -> np.ndarray
        Computes the output of the layer for a given input `x`.

    backward(grad: np.ndarray) -> tuple
        Computes gradients with respect to input, weights, and biases.

    apply(dw: np.ndarray, db: np.ndarray)
        Updates weights and biases using the given gradients and learning rate.
    """

    def __init__(self, in_dim, out_dim, lr=0.001):
        # Initialize weights and biases with uniform distribution in [-0.5, 0.5]
        self.weights = np.random.uniform(-0.5, 0.5, (out_dim, in_dim))
        self.biases  = np.random.uniform(-0.5, 0.5, out_dim)
        self.lr = lr  # Learning rate for updates

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Perform the forward pass through the dense layer.
        """
        self.x = x  # Store input for use in backward pass
        return self.weights.dot(x) + self.biases

    def backward(self, grad: np.ndarray):
        """
        Perform the backward pass (gradient computation).

        Parameters:
        -----------
        grad : np.ndarray
            Gradient from the next layer (with respect to this layer's output).

        Returns:
        --------
        tuple (dx, dw, db)
            dx: Gradient with respect to input x
            dw: Gradient with respect to weights
            db: Gradient with respect to biases
        """
        dx = self.weights.T.dot(grad)          # Gradient w.r.t input
        dw = np.outer(grad, self.x)            # Gradient w.r.t weights
        db = grad.copy()                       # Gradient w.r.t biases
        return dx, dw, db

    def apply(self, dw: np.ndarray, db: np.ndarray):
        """
        Apply the computed gradients to update weights and biases.
        """
        self.weights -= self.lr * dw
        self.biases  -= self.lr * db

# ------------------ Activation ------------------
class ReLU(Layer):
    """
    ReLU (Rectified Linear Unit) activation layer.

    Applies the element-wise function: f(x) = max(0, x)

    Methods:
    --------
    forward(x: np.ndarray) -> np.ndarray
        Applies the ReLU activation to the input.

    backward(grad: np.ndarray) -> np.ndarray
        Computes the gradient of the loss with respect to the input.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass for the ReLU activation.

        Parameters:
        -----------
        x : np.ndarray
            Input array.

        Returns:
        --------
        np.ndarray
            Output array where negative values are replaced with 0.
        """
        self.mask = x > 0  # Boolean mask where input > 0
        return x * self.mask  # Zero out negative elements

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        Backward pass for the ReLU activation.
        """
        return grad * self.mask  # Pass gradient only where input was > 0

# ------------------ HousePriceMLP ------------------
class HousePriceMLP:
    """
    A multi-layer perceptron (MLP) for predicting house prices, with support for MPI-based
    pipeline parallel training and optional GPU support using CuPy.

    Parameters:
    -----------
    lr : float
        Learning rate for all Dense layers.
    mpi_mgr : MPIManager
        Optional MPIManager instance for distributed training.
    pipeline : bool
        If True, enables pipeline parallelism across MPI ranks.

    Attributes:
    -----------
    full_layers : list
        The full sequence of layers for the complete model.
    layers : list
        The layers specific to this MPI rank (all layers in serial mode).
    loss_history : list
        Mean squared error (MSE) loss values recorded over training epochs.

    Methods:
    --------
    forward(x)              -- Forward pass through this rank's layer subset.
    backward(grad)          -- Backward pass with gradient updates.
    train_serial(X, y)      -- Serial training using full model on each rank.
    train_pipeline(X, y)    -- Pipeline-parallel training using micro-batching.
    save_weights(path)      -- Save model parameters to file.
    load_weights(path)      -- Load parameters and broadcast if necessary.
    predict(raw, X_min, X_max, y_min, y_max) -- Normalize, predict, then denormalize.
    get_loss_history()      -- Return stored MSE loss values.
    """
    def __init__(self, lr=1e-5, mpi_mgr=None, pipeline=False):
        self.manager = mpi_mgr
        self.rank    = mpi_mgr.rank if mpi_mgr else 0
        self.size    = mpi_mgr.size if mpi_mgr else 1

        # Set CUDA device to current rank (for GPU parallelism)
        cp.cuda.Device(self.rank).use()


        # Define the full model architecture (8 hidden layers + output)
        self.full_layers = [
            Dense(14,512,lr), ReLU(),
            Dense(512,256,lr), ReLU(),
            Dense(256,128,lr),  ReLU(),
            Dense(128,64,lr),  ReLU(),
            Dense(64,32, lr),  ReLU(),
            Dense(32,16, lr),  ReLU(),
            Dense(16,8, lr),  ReLU(),
            Dense(8,4, lr),  ReLU(),
            Dense(4,1, lr)
        ]
        self.pipeline = pipeline and (mpi_mgr is not None)
        if self.pipeline:
            # 1) group into Dense+ReLU “blocks”
            blocks = []
            i = 0
            while i < len(self.full_layers):
                blk = [ self.full_layers[i] ]
                # if there's a ReLU right after this Dense, pull it in
                if isinstance(self.full_layers[i], Dense) \
                   and i+1 < len(self.full_layers) \
                   and isinstance(self.full_layers[i+1], ReLU):
                    blk.append(self.full_layers[i+1])
                    i += 2
                else:
                    i += 1
                blocks.append(blk)

            num_blocks = len(blocks)
            num_stages = mpi_mgr.size

            # Ensure there are enough blocks for all stages
            if num_stages > num_blocks:
                raise ValueError(
                    f"Cannot pipeline across {num_stages} ranks: "
                    f"only {num_blocks} Dense-activation blocks available."
                )

            # 2) divide those blocks as evenly as possible
            per, rem = divmod(num_blocks, num_stages)
            start = 0
            my_blocks = []
            for r in range(num_stages):
                count = per + (1 if r < rem else 0)
                selected = blocks[start:start+count]
                if r == mpi_mgr.rank:
                    # flatten the list of lists into self.layers
                    self.layers = [ layer
                                    for blk in selected
                                    for layer in blk ]
                start += count

            # now every rank has self.layers containing at least one Dense
        else:
            # serial (or data-parallel?): everyone just keeps the full model
            self.layers = self.full_layers
            if mpi_mgr:
                mpi_mgr.broadcast_model(self)

        self.loss_history = []

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through this rank's subset of layers."""

        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass through this rank's subset of layers."""
        for layer in reversed(self.layers):
            if isinstance(layer, Dense):
                grad, dw, db = layer.backward(grad)
                layer.apply(dw, db)
            else:
                grad = layer.backward(grad)
        return grad

    def save_weights(self, filepath: str):
        """Save model parameters (weights and biases) to a JSON file."""

        params = {}
        for idx, layer in enumerate(self.full_layers):
            if isinstance(layer, Dense):
                params[f"layer_{idx}_weights"] = layer.weights.tolist()
                params[f"layer_{idx}_biases"]  = layer.biases.tolist()
        with open(filepath, 'w') as f:
            json.dump(params, f)

    def load_weights(self, filepath: str):
        """Load model parameters from a file and broadcast if needed."""

        with open(filepath, 'r') as f:
            data = json.load(f)
        for idx, layer in enumerate(self.full_layers):
            if isinstance(layer, Dense):
                key_w = f"layer_{idx}_weights"
                key_b = f"layer_{idx}_biases"
                if key_w in data and key_b in data:
                    layer.weights = np.array(data[key_w])
                    layer.biases  = np.array(data[key_b])
        if not self.pipeline and self.manager:
            self.manager.broadcast_model(self)

    def train_serial(self, X, y, epochs=20):
        """
        Train the model using serial (non-parallel) execution on a single device or MPI rank.

        Each training sample is processed individually using stochastic gradient descent (SGD).
        Gradients are computed and applied immediately after each forward pass.

        Parameters:
        -----------
        X : np.ndarray
            Input feature matrix of shape (num_samples, num_features).
        y : np.ndarray
            Target output values of shape (num_samples,) or (num_samples, 1).
        epochs : int
            Number of complete passes through the dataset.
        """
        for ep in range(epochs):
            start = time.time()

            # Shuffle data indices to ensure different sample order each epoch
            idxs = np.random.permutation(len(X))
            loss = 0.0

            for i in idxs:
                # Forward pass for a single sample
                diff = self.forward(X[i]) - y[i]

                # Accumulate squared error loss
                loss += float((diff**2).item())

                # Backward pass and immediate parameter update
                self.backward(2 * diff)

            # Compute mean squared error for the epoch
            mse = loss / len(X)
            self.loss_history.append(mse)

            # Only rank 0 prints the result (for distributed environments)
            if self.rank == 0:
                print(f"Serial Epoch {ep+1}: MSE={mse:.4f}, t={time.time()-start:.2f}s")


    def train_pipeline(self, X, y, epochs=20, micro_batch_size=8):
        """
        Train the model using pipeline parallelism and micro-batching across multiple MPI ranks.

        Each rank executes only part of the model (a subset of layers). During training,
        data flows forward stage-by-stage across the ranks (pipeline), and gradients flow
        back in reverse order.

        Parameters:
        -----------
        X : np.ndarray
            Input feature matrix of shape (num_samples, num_features).
        y : np.ndarray
            Target output values of shape (num_samples,) or (num_samples, 1).
        epochs : int
            Number of full passes over the dataset.
        micro_batch_size : int
            Number of examples processed per micro-batch in the pipeline.
        """

        # Setup MPI communication and rank info
        comm = self.manager.comm
        next_rank = self.rank + 1 if self.rank + 1 < self.size else None
        prev_rank = self.rank - 1 if self.rank - 1 >= 0 else None

        # Total number of micro-batches
        num_batches = (len(X) + micro_batch_size - 1) // micro_batch_size

        # Determine this stage's input dimension from the first Dense layer
        first_dense = next(layer for layer in self.layers if isinstance(layer, Dense))
        in_dim = first_dense.weights.shape[1]

        for ep in range(epochs):
            epoch_start = time.time()
            epoch_loss = 0.0

            for mb in range(num_batches):
                # Define batch range and extract input/output pairs
                start_idx = mb * micro_batch_size
                end_idx = min(start_idx + micro_batch_size, len(X))
                X_mb = X[start_idx:end_idx]
                y_mb = y[start_idx:end_idx]
                batch_size = len(X_mb)

                # --- Stage 0: Source Rank ---
                if self.rank == 0:
                    # Run local forward pass on all inputs in micro-batch
                    acts = np.vstack([self.forward(x) for x in X_mb])
                    # Send activations to next rank
                    comm.Send([acts, MPI.DOUBLE], dest=next_rank, tag=ep * num_batches + mb)
                    # Prepare buffer for receiving gradients back
                    grad_back = np.empty_like(acts)
                    comm.Recv([grad_back, MPI.DOUBLE], source=next_rank, tag=ep * num_batches + mb)
                    # Backpropagate received gradients for each sample
                    for g in grad_back:
                        self.backward(g)

                # --- Final Stage: Output Rank ---
                elif self.rank == self.size - 1:
                    # Receive activations from previous rank
                    buf = np.empty((batch_size, in_dim), dtype=float)
                    comm.Recv([buf, MPI.DOUBLE], source=prev_rank, tag=ep * num_batches + mb)
                    # Compute model outputs for each activation
                    outs = np.array([self.forward(x)[0] for x in buf])
                    # Compute errors
                    diffs = outs - y_mb
                    epoch_loss += np.sum(diffs ** 2)
                    grad_outs = (2 * diffs).reshape(-1, 1)
                    # Run backward pass to compute gradients w.r.t. previous stage
                    grad_prev = None
                    for g in grad_outs:
                        grad_prev = self.backward(np.array([g])[0])
                    # Replicate gradients for the batch and send them back
                    grad_batch = np.tile(grad_prev, (batch_size, 1))
                    comm.Send([grad_batch, MPI.DOUBLE], dest=prev_rank, tag=ep * num_batches + mb)

                # --- Intermediate Stage ---
                else:
                    # Receive activations from previous stage
                    buf_in = np.empty((batch_size, in_dim), dtype=float)
                    comm.Recv([buf_in, MPI.DOUBLE], source=prev_rank, tag=ep * num_batches + mb)
                    # Run forward pass on received activations
                    acts = np.vstack([self.forward(x) for x in buf_in])
                    # Send activations to next stage
                    comm.Send([acts, MPI.DOUBLE], dest=next_rank, tag=ep * num_batches + mb)
                    # Prepare to receive gradients
                    grad_back = np.empty_like(acts)
                    comm.Recv([grad_back, MPI.DOUBLE], source=next_rank, tag=ep * num_batches + mb)
                    # Run backward pass
                    grad_prev = None
                    for g in grad_back:
                        grad_prev = self.backward(g)
                    # Send gradients back to previous stage
                    grad_batch = np.tile(grad_prev, (batch_size, 1))
                    comm.Send([grad_batch, MPI.DOUBLE], dest=prev_rank, tag=ep * num_batches + mb)

            # Accumulate and report loss only from rank 0
            total_loss = comm.reduce(epoch_loss, op=MPI.SUM, root=0)
            if self.rank == 0:
                mse = total_loss / len(X)
                self.loss_history.append(mse)
                print(f"Pipeline Ep {ep+1}: MSE={mse:.4f}, time={time.time() - epoch_start:.3f}s")


    def predict(self, raw_input, X_min, X_max, y_min, y_max):
        """
        Generate a house price prediction for a single raw input example.

        This method first normalizes the input features using min-max scaling,
        runs a forward pass through the model, and then denormalizes the
        predicted output to return a value in the original price range.

        Parameters:
        -----------
        raw_input : list or np.ndarray
            A single input feature vector (unnormalized).
        X_min : list or np.ndarray
            Minimum values for each input feature (used for normalization).
        X_max : list or np.ndarray
            Maximum values for each input feature.
        y_min : float
            Minimum target value (e.g., lowest house price in training set).
        y_max : float
            Maximum target value.

        Returns:
        --------
        float
            The denormalized predicted house price.
        """
        # Normalize each feature using min-max scaling
        norm = np.array([
            (v - mn) / (mx - mn) if mx != mn else 0
            for v, mn, mx in zip(raw_input, X_min, X_max)
        ], dtype=float)

        # Run the normalized input through the network
        out = self.forward(norm)
        # Denormalize the output back to original price scale
        return out[0] * (y_max - y_min) + y_min


    def get_loss_history(self):
        """Return the accumulated loss history"""
        return self.loss_history

# ------------------ Data Loader ------------------
def load_data(csv_path: str):
    """
    Load and normalize a regression dataset from a CSV file.

    The function assumes that:
        - All columns except the last are input features (X).
        - The last column is the target variable (y).
        - Both X and y are normalized using min-max scaling.
        - It returns both the normalized data and the min/max values
          for later use in denormalization during prediction.

    Parameters:
    -----------
    csv_path : str
        Path to the CSV file containing the dataset.

    Returns:
    --------
    tuple :
        - Xn : np.ndarray
            Normalized feature matrix.
        - yn : np.ndarray
            Normalized target values.
        - Xm : list
            Minimum values of input features (for denormalization).
        - XM : list
            Maximum values of input features (for denormalization).
        - ym : float
            Minimum value of target variable.
        - yM : float
            Maximum value of target variable.
    """
    # Load dataset into a DataFrame
    df = pd.read_csv(csv_path)

    # Split features (X) and target (y)
    X = df.iloc[:, :-1]       # All columns except last
    y = df.iloc[:, -1].values # Last column as target

    # Compute min and max for each feature and target
    Xm, XM = X.min(), X.max() # Series
    ym, yM = y.min(), y.max() # Scalars

    # Normalize features and target using min-max scaling
    Xn = (X - Xm) / (XM - Xm)
    yn = (y - ym) / (yM - ym)

    # Return normalized data and original min/max for later scaling
    return Xn.values, yn, Xm.tolist(), XM.tolist(), ym, yM



"""
Command-Line Argument Guide
---------------------------

These flags control the behavior of the program at runtime. You can combine multiple flags
as needed to perform training, data/model pipelining, or prediction. It's important to note
that parallel options require launching the program using "mpirun.openmpi" and labeling the
number of processes

Arguments:
----------

--train
    Description: Run the training routine for the model.
    Type       : Flag (boolean)
    Usage      : Include this flag to train the model using either serial or pipeline mode.

--data_p
    Description: Enable data parallelism (used with --train).
    Type       : Flag (boolean)
    Usage      : Use this flag to activate data parallel training across MPI processes.

--model_p
    Description: Enable model pipeline parallelism (used with --train).
    Type       : Flag (boolean)
    Usage      : Use this flag to split the model across MPI ranks for pipeline-parallel training.

--predict
    Description: Run the prediction routine.
    Type       : Flag (boolean)
    Usage      : Include this flag to perform a prediction instead of training.

Examples:
---------

1. Train the model using MPI pipeline parallelism with 6 cores:
    mpirun.openmpi -np 6 python main.py --train --model_p

2. Train the model using serial training:
    python main.py --train

3. Predict using a trained model:
    python main.py --predict

"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train',    action='store_true')
    parser.add_argument('--data_p',   action='store_true')
    parser.add_argument('--model_p',  action='store_true')
    parser.add_argument('--predict',  action='store_true')
    args = parser.parse_args()

    # instantiate MPI Manager
    mpi_mgr = MPIManager()
    X, y, Xm, XM, ym, yM = load_data("house_prices.csv")
    # Splite data 80% train and 20% test
    split = int(0.8 * len(X)); Xtr, ytr = X[:split], y[:split]

    # Instantiate the model and set the learning rate + mpi manager + args
    model = HousePriceMLP(
        lr=1e-5,
        mpi_mgr=mpi_mgr,
        pipeline=args.model_p
    )
    # Check if there is a already pretrained weights file
    if os.path.exists("weights.json"):
        model.load_weights('weights.json')

    epochs = 20
    # change effect of runtime based on command line args
    # check for train command line argument
    if args.train:
        start_time = time.time()
        # check for model_p command line argument
        if args.model_p:
            Xb = mpi_mgr.comm.bcast(Xtr, root=0)
            yb = mpi_mgr.comm.bcast(ytr, root=0)
            model.train_pipeline(Xb, yb, epochs=epochs, micro_batch_size=20)

            end_time = time.time()
            exec_time = end_time-start_time
        # Data Parallelism hasn't been implemented for this excercise
        elif args.data_p:
            raise NotImplementedError
        
        else:
            model.train_serial(Xtr, ytr, epochs=epochs)

            end_time = time.time()
            exec_time = end_time-start_time
        
        # Regardless of if parallel or serial, then rank 0 saves weights at the end of training
        if mpi_mgr.rank == 0:
            model.save_weights("weights.json")
            print("Training completed in:", exec_time, " seconds")

    # check for predict command line argument
    elif args.predict and mpi_mgr.rank == 0:
        model.load_weights("weights.json")
        sample = [2000,3,2,10,1,2,1500,10.5,0.25,6,0,1,1,2010]
        price = model.predict(sample, Xm, XM, ym, yM)
        print(f"Predicted price: ${price:.2f}")
