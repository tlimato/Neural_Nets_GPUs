# Tyson Limato
# 6/1/2025
# Serial Code


# Notes 6/9/2025
# --- model parallel ---
# within 2 mpi ranks for 2 compute devices, within compute data chunk on certain layers then send at end of loop to rank 1 for receive given you are using send receive.
# then gpu 1x then 2x
# route comm through host and flag for nvlink (for direct comm.)
# Compare Comm time for host gpu host gpu vs. gpu to gpu
# matplot lib loss graphs for MSE and prediction loss, as well as other helpful graphs
# Metric vs freedom units (because international students)

# ------------------ Additional Resources ------------------
# https://wandb.ai/ayush-thakur/keras-dense/reports/Keras-Dense-Layer-How-to-Use-It-Correctly--Vmlldzo0MjAzNDY1




import random
import math
import json
import os.path
from typing import Literal
# External Libraries
import pandas as pd
from mpi4py import MPI

# Globals for Parallelism
comm =  MPI.COMM_WORLD
size = comm.size

# ------------------ Layer Base ------------------
class Layer:
    def forward(self, inputs):
        raise NotImplementedError

# ------------------ Dense Layer ------------------
class Dense(Layer):
    def __init__(self, input_size, output_size, learning_rate=0.001):
        self.weights = [[random.uniform(-0.5, 0.5) for _ in range(input_size)] for _ in range(output_size)]
        self.biases = [random.uniform(-0.5, 0.5) for _ in range(output_size)]
        self.learning_rate = learning_rate

    def get_params(self):
        return {
            'weights': self.weights,
            'biases': self.biases
        }

    def set_params(self, params):
        self.weights = params['weights']
        self.biases = params['biases']

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = []
        for i in range(len(self.weights)):
            weighted_sum = sum(w * x for w, x in zip(self.weights[i], inputs)) + self.biases[i]
            self.outputs.append(weighted_sum)
        return self.outputs

    def backward(self, dL_dout):
        MAX_GRAD = 1.0  # Clip gradient value
        dL_dinputs = [0] * len(self.inputs)
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                dL_dinputs[j] += dL_dout[i] * self.weights[i][j]

        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                grad = dL_dout[i] * self.inputs[j]
                grad = max(min(grad, MAX_GRAD), -MAX_GRAD)  # Clip
                self.weights[i][j] -= self.learning_rate * grad
            grad_b = max(min(dL_dout[i], MAX_GRAD), -MAX_GRAD)
            self.biases[i] -= self.learning_rate * grad_b

        return dL_dinputs
    
# ------------------ Activation Layer ------------------
class ReLU(Layer):
    def forward(self, inputs):
        self.inputs = inputs
        return [max(0, x) for x in inputs]

    def backward(self, dL_dout):
        return [dout if inp > 0 else 0 for inp, dout in zip(self.inputs, dL_dout)]

# ------------------ MSE Loss ------------------
def mean_squared_error(y_pred, y_true):
    return sum((yp - yt) ** 2 for yp, yt in zip(y_pred, y_true)) / len(y_pred)

# ------------------ Sigmoid ------------------
class Sigmoid(Layer):
    def forward(self, inputs):
        self.outputs = [1 / (1 + math.exp(-x)) for x in inputs]
        return self.outputs

    def backward(self, dL_dout):
        return [
            dout * out * (1 - out)
            for dout, out in zip(dL_dout, self.outputs)
        ]

# ------------------ Neural Network ------------------
class HousePriceMLP:
    def __init__(self, learning_rate=0.001):
        # Easily Abstract the layers for paralllel programming
        self.layers = [
            Dense(input_size=14, output_size=128, learning_rate=learning_rate),
            ReLU(),
            Dense(input_size=128, output_size=64, learning_rate=learning_rate),
            ReLU(),
            Dense(input_size=64, output_size=64, learning_rate=learning_rate),
            ReLU(),
            Dense(input_size=64, output_size=32, learning_rate=learning_rate),
            ReLU(),
            Dense(input_size=32, output_size=32, learning_rate=learning_rate),
            ReLU(),
            Dense(input_size=32, output_size=16, learning_rate=learning_rate),
            ReLU(),
            Dense(input_size=16, output_size=16, learning_rate=learning_rate),
            ReLU(),
            Dense(input_size=16, output_size=8, learning_rate=learning_rate),
            ReLU(),
            Dense(input_size=8, output_size=8, learning_rate=learning_rate),
            ReLU(),
            Dense(input_size=8, output_size=1, learning_rate=learning_rate),
        ]
        self.loss_history = []

    def save_weights(self, filepath):
        params = [layer.get_params() for layer in self.layers if isinstance(layer, Dense)]
        with open(filepath, 'w') as f:
            json.dump(params, f)

    def load_weights(self, filepath):
        with open(filepath, 'r') as f:
            params = json.load(f)
        dense_layers = [layer for layer in self.layers if isinstance(layer, Dense)]
        for layer, p in zip(dense_layers, params):
            layer.set_params(p)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dL_dout):
        for layer in reversed(self.layers):
            dL_dout = layer.backward(dL_dout)

    def train(self, X, y, epochs=10, parallel: Literal["false", "MPI", "MPI+CUDA"] = "false"):
       if parallel == "false": 
        for epoch in range(epochs):
            combined = list(zip(X, y))
            random.shuffle(combined)
            X[:], y[:] = zip(*combined)
            total_loss = 0

            for xi, yi in zip(X, y):
                pred = self.forward(xi)
                loss = (pred[0] - yi) ** 2
                dL_dpred = [2 * (pred[0] - yi)]  # ∂MSE/∂pred

                self.backward(dL_dpred)
                total_loss += loss

            mse = total_loss / len(X)
            self.loss_history.append(mse)
            print(f"Epoch {epoch+1}: MSE = {mse:.4f}")

       elif parallel == "MPI":
            raise NotImplementedError
       
       elif parallel == "MPI+CUDA":
            raise NotImplementedError
       else:
            raise Warning('Args: "parallel" Not Invoked Properly, please specify from list of valid args ["false", "MPI", "MPI+CUDA"]')
    
    def get_loss_history(self):
        return self.loss_history
        
# ------------------ Data Loader ------------------
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    X_df = df.iloc[:, :-1]
    y = df.iloc[:, -1].tolist()

    X_norm = (X_df - X_df.min()) / (X_df.max() - X_df.min())
    X = X_norm.values.tolist()

    y_min = min(y)
    y_max = max(y)
    y_norm = [(v - y_min) / (y_max - y_min) for v in y]
    return X, y_norm, y_min, y_max

# ------------------Predict Mode ------------------
# Predict on new data
def predict_new_house(model, raw_input, X_min, X_max, y_min, y_max):
    norm_input = [(val - min_val) / (max_val - min_val) if max_val != min_val else 0
                    for val, min_val, max_val in zip(raw_input, X_min, X_max)]
    norm_price = model.forward(norm_input)[0]
    return norm_price * (y_max - y_min) + y_min




# ------------------ Training and Inference Runtime ------------------
if __name__ == "__main__":
    model = HousePriceMLP(learning_rate=0.00001)
    if(os.path.isfile("weights.json")):
        model.load_weights("weights.json")
    # Load and split data
    X, y, y_min, y_max = load_data("house_prices.csv")

    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    # Train model
    model.train(X_train, y_train, epochs=5, parallel="false")
    model.save_weights("weights.json")

    # De-normalize predictions before evaluating
    test_preds = [model.forward(x)[0] * (y_max - y_min) + y_min for x in X_test]
    y_test_actual = [yt * (y_max - y_min) + y_min for yt in y_test]

    test_mse = mean_squared_error(test_preds, y_test_actual)
    print(f"\nTest MSE: {test_mse:.2f}")


    # Print training loss per epoch
    print("\nLoss history:")
    loss_table = model.get_loss_history()
    for i, loss in enumerate(loss_table):
        print(f"Epoch {i+1}: MSE = {loss:.4f}")

    # Example raw house input (must be 14 features)
    sample_house = [
        2000, 3, 2, 10, 1, 2, 1500, 10.5, 0.25, 6, 0, 1, 1, 2010
    ]

    # Get min and max from training data
    df = pd.read_csv("house_prices.csv")
    X_df = df.iloc[:, :-1]
    X_min = X_df.min().tolist()
    X_max = X_df.max().tolist()

    predicted_price = predict_new_house(model, sample_house, X_min, X_max, y_min, y_max)
    print(f"\nPredicted price for new house: ${predicted_price:.2f}") # canada, europe, australia, japan
