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
#   using: "conda create --name <env> --file package-list.txt" or executing 
#   the "setup_conda_env.sh" bash script. The large dependencies are below.
#       - mpi4py
#       - pandas
#       - numpy
#       - cupy (I used the Cuda 12x variant)
#
# Notes:
#   - C̶o̶m̶p̶l̶e̶t̶e̶ G̶P̶U̶ t̶e̶s̶t̶i̶n̶g̶
#   - i̶m̶p̶l̶e̶m̶e̶n̶t̶ g̶r̶a̶p̶h̶s̶ (̶l̶o̶s̶s̶,̶ t̶r̶a̶i̶n̶i̶n̶g̶ t̶i̶m̶e̶)̶
#   - c̶u̶r̶r̶e̶n̶c̶y̶ c̶o̶n̶v̶e̶r̶s̶i̶o̶n̶
#   - d̶e̶v̶e̶l̶o̶p̶ M̶P̶I̶ e̶x̶e̶r̶c̶i̶s̶e̶s̶
#   - d̶e̶v̶e̶l̶o̶p̶ C̶u̶P̶y̶ E̶x̶e̶r̶c̶i̶s̶e̶s̶
#   - m̶a̶k̶e̶ b̶a̶s̶h̶ s̶c̶r̶i̶p̶t̶ f̶o̶r̶ c̶o̶n̶d̶a̶ e̶n̶v̶i̶r̶o̶n̶m̶e̶n̶t̶
#   - T̶O̶D̶O̶:̶ C̶O̶N̶V̶E̶R̶T̶ T̶H̶I̶S̶ I̶N̶T̶O̶ A̶N̶ E̶X̶E̶R̶C̶I̶S̶E̶ F̶O̶R̶ S̶E̶N̶D̶I̶N̶G̶ F̶R̶O̶M̶ C̶P̶U̶ T̶O̶ G̶P̶U̶ a̶)̶ c̶o̶p̶y̶ d̶a̶t̶a̶ o̶f̶f̶ b̶)̶ c̶o̶m̶m̶ s̶e̶n̶d̶
#   - a̶d̶d̶ d̶e̶v̶i̶c̶e̶ i̶n̶f̶o̶ f̶o̶r̶ t̶h̶e̶ G̶P̶U̶ s̶e̶c̶t̶i̶o̶n̶
#   - s̶e̶g̶m̶e̶n̶t̶ c̶o̶d̶e̶ i̶n̶t̶o̶ m̶u̶l̶t̶i̶p̶l̶e̶ f̶i̶l̶e̶s̶
#   - update setup guide and create readme
#   - test on medicine bow once you get access (2 gpus)
#   - weights.json in its own folder and only load for predictions
#
# ------------------------------------------------------------
import pandas as pd
import time
import argparse
import os
import cupy.cuda.runtime as cuda_rt
from MLP_Model import HousePriceMLP
from mpiMGR import MPIManager


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



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train',    action='store_true')
    parser.add_argument('--gpu',      action='store_true')
    parser.add_argument('--data_p',   action='store_true')
    parser.add_argument('--model_p_MPI',  action='store_true')
    parser.add_argument('--predict',  action='store_true')
    parser.add_argument('--currency','-c', type=str,
                        choices=["USD", "EURO", "YEN", "GBP", "CAD", "INR", "CNY", "AUD", "MXN"])
    args = parser.parse_args()

    mpi_mgr = MPIManager()
    X, y, Xm, XM, ym, yM = load_data("house_prices.csv")
    split = int(0.8 * len(X))
    Xtr, ytr = X[:split], y[:split]

    model = None
    epochs = 20

    if args.train:
        # SERIAL MPI (no GPU)
        if args.model_p_MPI and not args.gpu:
            model = HousePriceMLP(lr=1e-5, mpi_mgr=mpi_mgr, pipeline=True)
            Xb = mpi_mgr.comm.bcast(Xtr, root=0)
            yb = mpi_mgr.comm.bcast(ytr, root=0)
            model.train_pipeline_MPI(Xb, yb, epochs=epochs, micro_batch_size=20)

            if mpi_mgr.rank == 0:
                model.plot_training_stats("pipeline_mpi_stats.png")

        # SERIAL GPU
        elif args.gpu and not args.model_p_MPI:
            model = HousePriceMLP(lr=1e-5, mpi_mgr=mpi_mgr,
                                  pipeline=False, use_gpu=True)
            free, _ = cuda_rt.memGetInfo()
            print(f"Start: GPU Memory Free: {free//1024**2} MB")
            model.data_parallel_gpu(Xtr, ytr, epochs=epochs, batch_size=64)
            free, _ = cuda_rt.memGetInfo()
            print(f"End: GPU Memory Free: {free//1024**2} MB")

            if mpi_mgr.rank == 0:
                model.plot_training_stats("serial_gpu_stats.png")

        # PIPELINE GPU
        elif args.model_p_MPI and args.gpu:
            model = HousePriceMLP(lr=1e-5, mpi_mgr=mpi_mgr,
                                  pipeline=True, use_gpu=True)
            Xb = mpi_mgr.comm.bcast(Xtr, root=0)
            yb = mpi_mgr.comm.bcast(ytr, root=0)
            model.train_pipeline_gpu(Xb, yb, epochs=epochs, micro_batch_size=20)

            if mpi_mgr.rank == 0:
                model.plot_training_stats("pipeline_gpu_stats.png")

        # SERIAL CPU
        else:
            model = HousePriceMLP(lr=1e-5, mpi_mgr=mpi_mgr, pipeline=False)
            model.train_serial(Xtr, ytr, epochs=epochs)

            if mpi_mgr.rank == 0:
                model.plot_training_stats("serial_cpu_stats.png")

        # Save weights on rank 0
        if mpi_mgr.rank == 0 and model:
            model.save_weights("weights.json")
            print("Training completed and weights.json saved.")

    # check for predict command line argument
    elif args.predict and mpi_mgr.rank == 0:
        if not args.currency:
            parser.error("--currency is required with --predict")

        model = HousePriceMLP(
            lr=1e-5,
            mpi_mgr=mpi_mgr,
            pipeline=False
        )
        model.load_weights("weights.json")

        sample = [2000,3,2,10,1,2,1500,10.5,0.25,6,0,1,1,2010]
        price = model.predict(sample, Xm, XM, ym, yM)

        # Conversions as of 6/18/2025
        match args.currency:
            case "USD":  print(f"Predicted price: ${price:,.2f}")
            case "EURO": print(f"Predicted price: €{(price * 0.87):,.2f}")
            case "YEN":  
                price = float(price)
                print(f"Predicted price: ¥{(price * 144.8):,.2f}")
            case "GBP":  print(f"Predicted price: £{(price * 0.7437):,.2f}")
            case "CAD":  print(f"Predicted price: ${(price * 1.366):,.2f}")
            case "INR":  print(f"Predicted price: ₹{(price * 86.52):,.2f}")
            case "CNY":  print(f"Predicted price: ¥{(price * 7.189):,.2f}")
            case "AUD":  print(f"Predicted price: A${(price * 1.534):,.2f}")
            case "MXN":  print(f"Predicted price: Mex${(price * 18.95):,.2f}")
