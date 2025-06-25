# Author      : Tyson Limato
# Date        : 2025-6-7
# File Name   : MLP_Model.py
from mpi4py import MPI
import pandas as pd
import numpy as np
import json
import time
import cupy as cp
from layers import Dense, DenseGPU, ReLU, ReLUGPU
import matplotlib.pyplot as plt

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
    train_pipeline_MPI(X, y)    -- Pipeline-parallel training using micro-batching.
    save_weights(path)      -- Save model parameters to file.
    load_weights(path)      -- Load parameters and broadcast if necessary.
    predict(raw, X_min, X_max, y_min, y_max) -- Normalize, predict, then denormalize.
    get_loss_history()      -- Return stored MSE loss values.
    """
    def __init__(self, lr=1e-5, mpi_mgr=None, pipeline=False, use_gpu=None):
        self.manager = mpi_mgr
        self.rank    = mpi_mgr.rank if mpi_mgr else 0
        self.size    = mpi_mgr.size if mpi_mgr else 1
        self.pipeline = pipeline and (mpi_mgr is not None)
        self.use_gpu = use_gpu

        # Select layer classes based on use_gpu
        DenseClass = DenseGPU if use_gpu else Dense
        ReLUClass  = ReLUGPU  if use_gpu else ReLU

        # If using GPU, assign this rank a GPU device
        if use_gpu:
            cp.cuda.Device(self.rank).use()

        # Define the full model architecture (8 hidden layers + output)
        self.full_layers = [
            DenseClass(14, 512, lr), ReLUClass(),
            DenseClass(512, 256, lr), ReLUClass(),
            DenseClass(256, 128, lr), ReLUClass(),
            DenseClass(128, 64, lr), ReLUClass(),
            DenseClass(64, 32, lr), ReLUClass(),
            DenseClass(32, 16, lr), ReLUClass(),
            DenseClass(16, 8, lr), ReLUClass(),
            DenseClass(8, 4, lr), ReLUClass(),
            DenseClass(4, 1, lr)
        ]

        if self.pipeline:
            # 1) Group into Dense+ReLU “blocks”
            blocks = []
            i = 0
            while i < len(self.full_layers):
                # create a block which is a list of layers the the first index is the current layer at i
                blk = [self.full_layers[i]]
                # Check if the layer i is a dense class, index i + 1 is < length, and layer i + 1 is a ReLU Class. If so append the i+1 layer to the block
                if isinstance(self.full_layers[i], DenseClass) \
                and i+1 < len(self.full_layers) \
                and isinstance(self.full_layers[i+1], ReLUClass):
                    blk.append(self.full_layers[i+1])
                    # increment "i" by 2 so that that layers are double added to blocks
                    i += 2
                else:
                    # increment "i" normally if current layer at "self.full_layers[i]" isn't a Dense Layer (starting index could be a Relu)
                    i += 1
                # append the block "blk" to the blocks list which contains the layer blocks for our pipeline
                blocks.append(blk)

            num_blocks = len(blocks)
            # num_stages is the number of MPI ranks and/or devices we have to parallelize our data across.  the number of stages cannot be greater than the number of blocks. 
            num_stages = self.size

            if num_stages > num_blocks:
                raise ValueError(
                    f"Cannot pipeline across {num_stages} ranks: "
                    f"only {num_blocks} Dense-activation blocks available."
                )

            # 2) Divide blocks evenly
            per, rem = divmod(num_blocks, num_stages) # allows us to easily get the quotient and remainder of our num_blocks / num_stages
            start = 0
            for r in range(num_stages):
                count = per + (1 if r < rem else 0)
                selected = blocks[start:start+count]
                if r == self.rank:
                    self.layers = [layer for blk in selected for layer in blk]
                start += count
        else:
            # Serial or data-parallel
            self.layers = self.full_layers
            if mpi_mgr:
                mpi_mgr.broadcast_model(self)

        self.loss_history = []
        self.time_history = []
        

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through this rank's subset of layers."""

        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out


    def backward(self, grad):
        for layer in reversed(self.layers):
            # CPU or GPU dense?
            if isinstance(layer, (Dense, DenseGPU)):
                # both Dense and DenseGPU.backward return (dx, dw, db)
                dx, dw, db = layer.backward(grad)
                layer.apply(dw, db)
                grad = dx
            else:
                # activation layers (ReLU/ReLUGPU) just map grad→grad
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
            epoch_time = time.time() - start
            self.time_history.append(epoch_time)
            # Only rank 0 prints the result (for distributed environments)
            if self.rank == 0:
                print(f"Serial Epoch {ep+1}: MSE={mse:.4f}, t={epoch_time:.2f}s")

    def data_parallel_gpu(self, X, y, epochs=20, batch_size=64):
        # Get number of GPUs available
        num_gpus = cp.cuda.runtime.getDeviceCount()
        print(f"Number of GPUs: {num_gpus}")

        # Print info for each GPU
        for i in range(num_gpus):
            props = cp.cuda.runtime.getDeviceProperties(i)
            print(f"\nGPU {i}: {props['name'].decode()}")
            print(f"  Compute Capability: {props['major']}.{props['minor']}")
            print(f"  Memory: {props['totalGlobalMem'] / 1e6:.2f} MB")
            print(f"  Multiprocessors: {props['multiProcessorCount']}\n")
        # 1) Move all data to GPU once
        X_gpu = cp.asarray(X)      # shape (N, 14)
        y_gpu = cp.asarray(y)      # shape (N,)

        for ep in range(epochs):
            # 2) shuffle indices on GPU
            idxs = cp.random.permutation(len(X_gpu))
            total_loss = 0.0
            epoch_start = time.time()
            # 3) process in batches of batch_size
            for start in range(0, len(X_gpu), batch_size):
                batch_idx = idxs[start:start+batch_size]
                Xb = X_gpu[batch_idx]      # shape (B, 14)
                yb = y_gpu[batch_idx]      # shape (B,)

                # 4) Forward pass for entire batch
                out = Xb
                for layer in self.layers:
                    # each layer.forward must accept a 2D array (B, …)
                    out = layer.forward(out)
                preds = out.reshape(-1)     # shape (B,)

                # 5) Compute loss & gradient
                diff = preds - yb           # (B,)
                total_loss += float(cp.sum(diff**2))
                grad = (2 * diff / batch_size).reshape(-1, 1)  
                                        # shape (B,1): gradient per sample

                # 6) Backward pass for the batch
                for layer in reversed(self.layers):
                    if isinstance(layer, DenseGPU):
                        dx, dw, db = layer.backward(grad)
                        layer.apply(dw, db)
                        grad = dx               # pass batch‐grad down
                    else:
                        grad = layer.backward(grad)

            # 7) report
            mse = total_loss / len(X_gpu)
            # Compute mean squared error for the epoch
            self.loss_history.append(mse)
            epoch_time = time.time() - epoch_start
            self.time_history.append(epoch_time)
            print(f"[GPU-Batch] Epoch {ep+1}: MSE={mse:.4f}, t={epoch_time:.2f}s")


    # START OF SLIDE 14: Pipeline Model Parallelism with MPI
    def train_pipeline_MPI(self, X, y, epochs=20, micro_batch_size=8):
        """
        Train the model using pipeline parallelism and micro-batching across multiple MPI ranks.
        
        Conceptually, we split the network into consecutive "stages" (each MPI rank holds one stage).
        We then send small groups of examples (micro-batches) forward through the stages, one after the other.
        Once the final stage has computed its loss, gradients flow backward stage-by-stage in reverse order.
        """

        # -------------------------------------------------------------
        # 0) Set up MPI communication → Slide 20: MPI Concepts Applied
        # -------------------------------------------------------------
        comm = self.manager.comm           # MPI communicator (handles send/recv)
        next_rank = self.rank + 1 if self.rank + 1 < self.size else None
        prev_rank = self.rank - 1 if self.rank - 1 >= 0 else None

        # -------------------------------------------------------------
        # 1) Determine how many micro-batches we need → Slide 17: Micro-Batching for Pipelining Efficiency
        # -------------------------------------------------------------
        # e.g. if 100 samples and micro_batch_size=8, we get 13 batches of varying size
        num_batches = (len(X) + micro_batch_size - 1) // micro_batch_size

        # -------------------------------------------------------------
        # 2) Figure out activation dimension for communication buffers → Slide 19: Memory Management and Buffer Allocation
        # -------------------------------------------------------------
        first_dense = next(layer for layer in self.layers if isinstance(layer, Dense))
        in_dim = first_dense.weights.shape[1]

        # -------------------------------------------------------------
        # 3) Epoch loop: repeat forward/backward passes
        # -------------------------------------------------------------
        for ep in range(epochs):
            epoch_start = time.time()
            epoch_loss = 0.0

            # ---------------------------------------------------------
            # 4) Micro-batch loop → Slide 17: Micro-Batching
            # ---------------------------------------------------------
            for mb in range(num_batches):
                # 4a) Compute slice indices for this micro-batch
                start_idx = mb * micro_batch_size
                end_idx   = min(start_idx + micro_batch_size, len(X))
                X_mb = X[start_idx:end_idx]      # shape (batch_size, features)
                y_mb = y[start_idx:end_idx]      # shape (batch_size,)
                batch_size = len(X_mb)

                # -----------------------------------------------------
                # 4b) Stage 0 (source rank) – produce and send activations → Slide 16: Forward Pass – Rank 0
                # -----------------------------------------------------
                if self.rank == 0:
                    # Run forward pass on each sample locally
                    # stacks to shape (batch_size, activation_dim)
                    acts = np.vstack([self.forward(x) for x in X_mb])

                    # Send these activations to the next MPI rank
                    # MPI.Send works on NumPy arrays (host memory)
                    comm.Send([acts, MPI.DOUBLE],
                            dest=next_rank,
                            tag=ep * num_batches + mb)

                    # Prepare a buffer to receive gradients back from next rank
                    grad_back = np.empty_like(acts)
                    comm.Recv([grad_back, MPI.DOUBLE],
                            source=next_rank,
                            tag=ep * num_batches + mb)

                    # Back-propagate each gradient through local layers
                    for g in grad_back:
                        self.backward(g)

                # -----------------------------------------------------
                # 4c) Final Stage (sink rank) – compute loss & initial grads → Slide 15 & Slide 16: Sync Rank Logic
                # -----------------------------------------------------
                elif self.rank == self.size - 1:
                    # Receive activations from previous rank
                    buf = np.empty((batch_size, in_dim), dtype=float)
                    comm.Recv([buf, MPI.DOUBLE],
                            source=prev_rank,
                            tag=ep * num_batches + mb)

                    # Compute outputs and loss locally
                    outs = np.array([self.forward(x)[0] for x in buf])
                    diffs = outs - y_mb
                    epoch_loss += np.sum(diffs ** 2)

                    # Compute gradients dL/dout = 2 * (pred - true)
                    grad_outs = (2 * diffs).reshape(-1, 1)

                    # Backward pass to get gradient w.r.t. inputs of this stage
                    grad_prev = None
                    for g in grad_outs:
                        grad_prev = self.backward(np.array([g])[0])

                    # Replicate last gradient for entire micro-batch
                    grad_batch = np.tile(grad_prev, (batch_size, 1))

                    # Send gradients back to previous rank
                    comm.Send([grad_batch, MPI.DOUBLE],
                            dest=prev_rank,
                            tag=ep * num_batches + mb)

                # -----------------------------------------------------
                # 4d) Intermediate Stages – relay activations & gradients → Slide 16 and 17: Relay Logic
                # -----------------------------------------------------
                else:
                    # Receive activations from previous stage
                    buf_in = np.empty((batch_size, in_dim), dtype=float)
                    comm.Recv([buf_in, MPI.DOUBLE],
                            source=prev_rank,
                            tag=ep * num_batches + mb)

                    # Forward through local layers
                    acts = np.vstack([self.forward(x) for x in buf_in])

                    # Send to next rank
                    comm.Send([acts, MPI.DOUBLE],
                            dest=next_rank,
                            tag=ep * num_batches + mb)

                    # Receive gradients back from next rank
                    grad_back = np.empty_like(acts)
                    comm.Recv([grad_back, MPI.DOUBLE],
                            source=next_rank,
                            tag=ep * num_batches + mb)

                    # Backward through local layers
                    grad_prev = None
                    for g in grad_back:
                        grad_prev = self.backward(g)

                    # Send gradient back to previous rank
                    grad_batch = np.tile(grad_prev, (batch_size, 1))
                    comm.Send([grad_batch, MPI.DOUBLE],
                            dest=prev_rank,
                            tag=ep * num_batches + mb)

            # -------------------------------------------------------------
            # 5) End of epoch: gather total loss and record time → Slide 20: MPI Reduce
            # -------------------------------------------------------------
            total_loss = comm.reduce(epoch_loss, op=MPI.SUM, root=0)
            if self.rank == 0:
                mse = total_loss / len(X)
                self.loss_history.append(mse)

                # record epoch duration
                epoch_time = time.time() - epoch_start
                self.time_history.append(epoch_time)

                print(f"Pipeline Ep {ep+1}: MSE={mse:.4f}, time={epoch_time:.3f}s")


    def train_pipeline_gpu(self, X, y, epochs=20, micro_batch_size=8):
        """
        Pipeline-parallel training on GPU. Data is processed in small "micro-batches"
        that flow forward through consecutive model stages (across MPI ranks), then
        gradients flow backward in reverse order. Activations and gradients are
        marshaled between GPUs by copying to/from host (NumPy) buffers and sending
        via MPI, while all layer compute stays on the GPU via CuPy.
        """
        # Get number of GPUs available
        num_gpus = cp.cuda.runtime.getDeviceCount()
        print(f"Number of GPUs: {num_gpus}")

        # Print info for each GPU
        for i in range(num_gpus):
            props = cp.cuda.runtime.getDeviceProperties(i)
            print(f"\nGPU {i}: {props['name'].decode()}")
            print(f"  Compute Capability: {props['major']}.{props['minor']}")
            print(f"  Memory: {props['totalGlobalMem'] / 1e6:.2f} MB")
            print(f"  Multiprocessors: {props['multiProcessorCount']}")
        # If there's only one MPI rank, just do serial GPU training instead
        if self.size == 1:
            return self.data_parallel_gpu(X, y, epochs=epochs)

        # Move entire dataset into GPU memory once (CuPy arrays live on the device)
        X_gpu = cp.asarray(X)
        y_gpu = cp.asarray(y)

        # Extract MPI communicator and rank information
        comm      = self.manager.comm
        rank      = self.rank      # this process's rank (0..size-1)
        size      = self.size      # total number of ranks
        next_rank = rank + 1 if rank + 1 < size else None
        prev_rank = rank - 1 if rank - 1 >= 0      else None

        # Compute how many micro-batches fit into the dataset
        num_batches = (len(X_gpu) + micro_batch_size - 1) // micro_batch_size

        # Figure out the dimension of activations sent between stages
        first_dense = next(l for l in self.layers if isinstance(l, DenseGPU))
        in_dim = first_dense.weights.shape[1]

        for ep in range(epochs):
            epoch_start = time.time()
            epoch_loss = 0.0

            # Loop over micro-batches
            for mb in range(num_batches):
                # Determine slice indices for this micro-batch
                start_idx = mb * micro_batch_size
                end_idx   = min(start_idx + micro_batch_size, len(X_gpu))
                batch_size = end_idx - start_idx

                # ----- Stage 0: the "source" rank produces activations -----
                if rank == 0:
                    # 1) Forward on GPU for this micro-batch:
                    #    out = model.forward(x) for each sample of X_gpu[start_idx:end_idx]
                    acts_gpu = cp.vstack([
                        self.forward(x) 
                        for x in X_gpu[start_idx:end_idx]
                    ])
                    # 2) Copy activations off the GPU into a NumPy array
                    acts = cp.asnumpy(acts_gpu)
                    # 3) Send them via MPI to the next rank
                    comm.Send([acts, MPI.DOUBLE], dest=next_rank,
                            tag=ep * num_batches + mb)

                    # 4) Receive gradients from next rank back into a NumPy buffer
                    grad_back = np.empty_like(acts)
                    comm.Recv([grad_back, MPI.DOUBLE],
                            source=next_rank, tag=ep * num_batches + mb)
                    # 5) Move those grads onto GPU and apply backward
                    grad_back_gpu = cp.asarray(grad_back)
                    for g in grad_back_gpu:
                        self.backward(g)

                # ----- Final Stage: the sync rank computes loss and initial grads -----
                elif rank == size - 1:
                    # 1) Receive activations from previous rank (NumPy buffer)
                    buf = np.empty((batch_size, in_dim), dtype=float)
                    comm.Recv([buf, MPI.DOUBLE],
                            source=prev_rank, tag=ep * num_batches + mb)
                    # 2) Copy to GPU
                    buf_gpu = cp.asarray(buf)

                    # 3) Forward + compute per-sample loss on GPU
                    outs_gpu = cp.array([
                        self.forward(x)[0] 
                        for x in buf_gpu
                    ])
                    diffs = cp.asnumpy(outs_gpu) - np.array(
                        y[start_idx:end_idx]
                    )
                    epoch_loss += np.sum(diffs**2)

                    # 4) Compute gradients (2 * diff), backward on GPU
                    grad_prev_gpu = None
                    for d in diffs:
                        grad_prev_gpu = self.backward(cp.asarray([2 * d]))
                    # 5) Tile that last gradient for the whole batch
                    grad_batch_gpu = cp.tile(grad_prev_gpu, (batch_size, 1))
                    # 6) Send gradients back to previous rank (via NumPy)
                    comm.Send([cp.asnumpy(grad_batch_gpu), MPI.DOUBLE],
                            dest=prev_rank, tag=ep * num_batches + mb)

                # ----- Intermediate Stages: receive, forward, send, then backward -----
                else:
                    # 1) Receive activations from prev rank (NumPy), move to GPU
                    buf_in = np.empty((batch_size, in_dim), dtype=float)
                    comm.Recv([buf_in, MPI.DOUBLE],
                            source=prev_rank, tag=ep * num_batches + mb)
                    acts_gpu = cp.vstack([
                        self.forward(x)
                        for x in cp.asarray(buf_in)
                    ])
                    # 2) Send these activations on to next rank
                    comm.Send([cp.asnumpy(acts_gpu), MPI.DOUBLE],
                            dest=next_rank, tag=ep * num_batches + mb)

                    # 3) Receive gradient from next rank, move to GPU
                    grad_back = np.empty_like(buf_in)
                    comm.Recv([grad_back, MPI.DOUBLE],
                            source=next_rank, tag=ep * num_batches + mb)
                    grad_back_gpu = cp.asarray(grad_back)

                    # 4) Backward on GPU through local layers
                    grad_prev_gpu = None
                    for g in grad_back_gpu:
                        grad_prev_gpu = self.backward(g)
                    # 5) Tile and send grads back to previous rank
                    grad_batch_gpu = cp.tile(grad_prev_gpu, (batch_size, 1))
                    comm.Send([cp.asnumpy(grad_batch_gpu), MPI.DOUBLE],
                            dest=prev_rank, tag=ep * num_batches + mb)

            # End of epoch: collect total loss across ranks and record time
            total_loss = comm.reduce(epoch_loss, op=MPI.SUM, root=0)
            if rank == 0:
                mse = total_loss / len(X_gpu)
                self.loss_history.append(mse)
                epoch_time = time.time() - epoch_start
                self.time_history.append(epoch_time)
                print(f"[GPU-Pipeline] Epoch {ep+1}: MSE={mse:.4f}, time={epoch_time:.2f}s")


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

    def plot_training_stats(self, filename: str = 'training_stats.png'):
        """
        Uses matplotlib to plot self.loss_history and self.time_history
        (one curve each) against epoch number, and saves to `filename`.
        """
        

        epochs = list(range(1, len(self.loss_history) + 1))
        fig, ax1 = plt.subplots(figsize=(8,5))

        # Plot MSE on left axis
        ax1.plot(epochs, self.loss_history,
                label='MSE', linestyle='-', marker='o')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Error (MSE)')
        ax1.tick_params(axis='y')

        # Plot time on right axis
        ax2 = ax1.twinx()
        ax2.plot(epochs, self.time_history,
                label='Time (s)', linestyle='--', marker='x')
        ax2.set_ylabel('Epoch Time (s)')
        ax2.tick_params(axis='y')

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right',
                fontsize='small')

        plt.title('Training Error & Time per Epoch')
        fig.tight_layout()
        fig.savefig(filename)
        plt.close(fig)


    def get_loss_history(self):
        """Return the accumulated loss history"""
        return self.loss_history
    