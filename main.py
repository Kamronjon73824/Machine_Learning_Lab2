import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


# =========================================================
# TASK 1: DATASET PREPARATION
# =========================================================

# 1. Load dataset
digits = load_digits()

X = digits.data
y = digits.target

print("Original X shape:", X.shape)
print("Original y shape:", y.shape)

# 2. Scale input values from 0-16 to 0-1
X = X / 16.0

print("Min pixel value:", X.min())
print("Max pixel value:", X.max())

# 3. One-hot encode labels
num_classes = 10
y_onehot = np.eye(num_classes)[y]

print("One-hot shape:", y_onehot.shape)
print("Example label:", y[0])
print("Example one-hot:", y_onehot[0])

# 4. Split dataset: 70% training, 30% testing
X_train, X_test, y_train, y_test, y_train_labels, y_test_labels = train_test_split(
    X,
    y_onehot,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print("\nTrain X:", X_train.shape)
print("Train y:", y_train.shape)
print("Test X:", X_test.shape)
print("Test y:", y_test.shape)


# =========================================================
# TASK 2: NEURAL NETWORK ARCHITECTURE
# =========================================================

def initialize_parameters(width, seed=42):
    rng = np.random.default_rng(seed)

    params = {
        "W1": rng.normal(0, 0.01, size=(64, width)),
        "b1": np.zeros(width),

        "W2": rng.normal(0, 0.01, size=(width, width)),
        "b2": np.zeros(width),

        "W3": rng.normal(0, 0.01, size=(width, 10)),
        "b3": np.zeros(10)
    }

    return params


# Check both required hidden widths
for width in [16, 64]:
    params = initialize_parameters(width)

    print(f"\nNetwork with hidden width W = {width}")

    print("W1 shape:", params["W1"].shape)
    print("b1 shape:", params["b1"].shape)

    print("W2 shape:", params["W2"].shape)
    print("b2 shape:", params["b2"].shape)

    print("W3 shape:", params["W3"].shape)
    print("b3 shape:", params["b3"].shape)

# =========================================================
# TASK 3: FORWARD PROPAGATION
# =========================================================

def relu(x):
    return np.maximum(0, x)


def softmax(x):
    # Numerical stability
    x = x - np.max(x, axis=1, keepdims=True)

    exp_x = np.exp(x)

    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def forward(X, params):
    # Layer 1
    Z1 = X @ params["W1"] + params["b1"]
    A1 = relu(Z1)

    # Layer 2
    Z2 = A1 @ params["W2"] + params["b2"]
    A2 = relu(Z2)

    # Output layer
    Z3 = A2 @ params["W3"] + params["b3"]
    probabilities = softmax(Z3)

    cache = {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2,
        "Z3": Z3,
        "probabilities": probabilities
    }

    return probabilities, cache


# ---------------------------------------------------------
# Test forward propagation
# ---------------------------------------------------------

params_test = initialize_parameters(width=16)

# Use first 32 training samples as one batch
X_batch = X_train[:32]

probabilities, cache = forward(X_batch, params_test)

print("\nTASK 3 - FORWARD PROPAGATION")

print("Input batch shape:", X_batch.shape)
print("Output shape:", probabilities.shape)

print("\nFirst probability row:")
print(probabilities[0])

print("\nSum of first probability row:")
print(np.sum(probabilities[0]))

# Check all rows sum to 1
row_sums = np.sum(probabilities, axis=1)

print("\nAll rows sum to 1:",
      np.allclose(row_sums, 1.0, atol=1e-6))

# =========================================================
# TASK 4: CROSS-ENTROPY LOSS
# =========================================================

def cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-12

    y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)

    loss = -np.mean(
        np.sum(y_true * np.log(y_pred), axis=1)
    )

    return loss


# Use the same 32-sample batch
y_batch = y_train[:32]

loss = cross_entropy_loss(y_batch, probabilities)

print("\nTASK 4 - CROSS-ENTROPY LOSS")
print("Untrained network loss:", loss)
print("Expected approximately:", np.log(10))

# =========================================================
# TASK 5: BACKPROPAGATION
# =========================================================

def backward(X, y_true, params, cache):
    batch_size = X.shape[0]

    A1 = cache["A1"]
    A2 = cache["A2"]
    Z1 = cache["Z1"]
    Z2 = cache["Z2"]
    probabilities = cache["probabilities"]

    # -----------------------------------------------------
    # Output layer
    # Softmax + Cross-Entropy derivative
    # -----------------------------------------------------
    dZ3 = (probabilities - y_true) / batch_size

    dW3 = A2.T @ dZ3
    db3 = np.sum(dZ3, axis=0)

    # -----------------------------------------------------
    # Hidden layer 2
    # -----------------------------------------------------
    dA2 = dZ3 @ params["W3"].T

    # ReLU derivative
    dZ2 = dA2 * (Z2 > 0)

    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0)

    # -----------------------------------------------------
    # Hidden layer 1
    # -----------------------------------------------------
    dA1 = dZ2 @ params["W2"].T

    # ReLU derivative
    dZ1 = dA1 * (Z1 > 0)

    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0)

    gradients = {
        "W1": dW1,
        "b1": db1,
        "W2": dW2,
        "b2": db2,
        "W3": dW3,
        "b3": db3
    }

    return gradients


# ---------------------------------------------------------
# Run backward propagation
# ---------------------------------------------------------

params_gradient = initialize_parameters(width=16, seed=42)

X_gradient = X_train[:16]
y_gradient = y_train[:16]

probabilities_gradient, cache_gradient = forward(
    X_gradient,
    params_gradient
)

gradients = backward(
    X_gradient,
    y_gradient,
    params_gradient,
    cache_gradient
)

print("\nTASK 5 - BACKPROPAGATION")

print("dW1 shape:", gradients["W1"].shape)
print("db1 shape:", gradients["b1"].shape)

print("dW2 shape:", gradients["W2"].shape)
print("db2 shape:", gradients["b2"].shape)

print("dW3 shape:", gradients["W3"].shape)
print("db3 shape:", gradients["b3"].shape)


# =========================================================
# NUMERICAL GRADIENT CHECKING
# =========================================================

def gradient_check(
    X,
    y_true,
    params,
    gradients,
    num_checks=10,
    epsilon=1e-5,
    seed=123
):
    rng = np.random.default_rng(seed)

    # We check weights only, as required by the lab
    weight_names = ["W1", "W2", "W3"]

    # Build list of all possible weight positions
    all_positions = []

    for name in weight_names:
        rows, cols = params[name].shape

        for i in range(rows):
            for j in range(cols):
                all_positions.append((name, i, j))

    # Select 10 random weights
    selected_indices = rng.choice(
        len(all_positions),
        size=num_checks,
        replace=False
    )

    passed = 0

    print("\nGRADIENT CHECK")

    for check_number, index in enumerate(selected_indices, start=1):
        name, i, j = all_positions[index]

        original_value = params[name][i, j]

        # -----------------------------------------------
        # Loss with weight + epsilon
        # -----------------------------------------------
        params[name][i, j] = original_value + epsilon

        probs_plus, _ = forward(X, params)

        loss_plus = cross_entropy_loss(
            y_true,
            probs_plus
        )

        # -----------------------------------------------
        # Loss with weight - epsilon
        # -----------------------------------------------
        params[name][i, j] = original_value - epsilon

        probs_minus, _ = forward(X, params)

        loss_minus = cross_entropy_loss(
            y_true,
            probs_minus
        )

        # Restore original value
        params[name][i, j] = original_value

        # -----------------------------------------------
        # Numerical gradient
        # -----------------------------------------------
        numerical_gradient = (
            loss_plus - loss_minus
        ) / (2 * epsilon)

        # Analytical gradient from backpropagation
        analytical_gradient = gradients[name][i, j]

        # Relative error
        denominator = max(
            1e-12,
            abs(numerical_gradient) + abs(analytical_gradient)
        )

        relative_error = abs(
            numerical_gradient - analytical_gradient
        ) / denominator

        if relative_error < 1e-6:
            status = "PASS"
            passed += 1
        else:
            status = "FAIL"

        print(
            f"{check_number}: "
            f"{name}[{i},{j}] "
            f"analytic={analytical_gradient:.10f} "
            f"numerical={numerical_gradient:.10f} "
            f"relative_error={relative_error:.2e} "
            f"{status}"
        )

    print(f"\nPassed: {passed}/{num_checks}")

    if passed >= 9:
        print("Gradient check PASSED")
    else:
        print("Gradient check FAILED")


gradient_check(
    X_gradient,
    y_gradient,
    params_gradient,
    gradients
)

# =========================================================
# TASK 6: TRAINING
# =========================================================

import os
import matplotlib.pyplot as plt


def update_parameters(params, gradients, learning_rate):
    params["W1"] -= learning_rate * gradients["W1"]
    params["b1"] -= learning_rate * gradients["b1"]

    params["W2"] -= learning_rate * gradients["W2"]
    params["b2"] -= learning_rate * gradients["b2"]

    params["W3"] -= learning_rate * gradients["W3"]
    params["b3"] -= learning_rate * gradients["b3"]


def calculate_accuracy(X, labels, params):
    probabilities, _ = forward(X, params)

    predictions = np.argmax(probabilities, axis=1)

    accuracy = np.mean(predictions == labels)

    return accuracy


def train_network(
    X_train,
    y_train,
    width,
    learning_rate,
    epochs=50,
    batch_size=32,
    seed=42
):
    params = initialize_parameters(width, seed=seed)

    rng = np.random.default_rng(seed)

    losses = []

    number_of_samples = X_train.shape[0]

    print("\n========================================")
    print(
        f"Training: width={width}, "
        f"learning_rate={learning_rate}"
    )
    print("========================================")

    for epoch in range(epochs):

        # ---------------------------------------------
        # Shuffle training data
        # ---------------------------------------------
        indices = rng.permutation(number_of_samples)

        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        # ---------------------------------------------
        # Mini-batch training
        # ---------------------------------------------
        for start in range(0, number_of_samples, batch_size):

            end = start + batch_size

            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            # Forward propagation
            probabilities, cache = forward(
                X_batch,
                params
            )

            # Backpropagation
            gradients = backward(
                X_batch,
                y_batch,
                params,
                cache
            )

            # Gradient descent
            update_parameters(
                params,
                gradients,
                learning_rate
            )

        # ---------------------------------------------
        # Calculate loss after the epoch
        # ---------------------------------------------
        train_probabilities, _ = forward(
            X_train,
            params
        )

        epoch_loss = cross_entropy_loss(
            y_train,
            train_probabilities
        )

        losses.append(epoch_loss)

        print(
            f"Epoch {epoch + 1:02d}/{epochs} "
            f"- Loss: {epoch_loss:.6f}"
        )

    return params, losses


# =========================================================
# RUN ALL 6 EXPERIMENTS
# =========================================================

learning_rates = [0.001, 0.01, 0.1]
widths = [16, 64]

experiment_results = {}

for width in widths:

    for learning_rate in learning_rates:

        trained_params, losses = train_network(
            X_train,
            y_train,
            width=width,
            learning_rate=learning_rate,
            epochs=50,
            batch_size=32
        )

        test_accuracy = calculate_accuracy(
            X_test,
            y_test_labels,
            trained_params
        )

        experiment_name = (
            f"W={width}, lr={learning_rate}"
        )

        experiment_results[experiment_name] = {
            "losses": losses,
            "accuracy": test_accuracy
        }

        print(
            f"\nFinal test accuracy "
            f"({experiment_name}): "
            f"{test_accuracy * 100:.2f}%"
        )


# =========================================================
# PLOT LOSS CURVES
# =========================================================

os.makedirs("results", exist_ok=True)

plt.figure(figsize=(10, 6))

for experiment_name, result in experiment_results.items():

    plt.plot(
        range(1, 51),
        result["losses"],
        label=experiment_name
    )

plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("Training Loss Curves")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "results/loss_curves.png",
    dpi=300
)

plt.close()

print("\nLoss curves saved to:")
print("results/loss_curves.png")


# =========================================================
# FINAL RESULTS SUMMARY
# =========================================================

print("\n========================================")
print("FINAL TEST ACCURACIES")
print("========================================")

for experiment_name, result in experiment_results.items():

    print(
        f"{experiment_name}: "
        f"{result['accuracy'] * 100:.2f}%"
    )

# =========================================================
# TASK 7: FORWARD / BACKWARD TIMING
# =========================================================

import time


def measure_timing(width, runs=100):
    params = initialize_parameters(width, seed=42)

    # Use one mini-batch of 32 samples
    X_batch = X_train[:32]
    y_batch = y_train[:32]

    # Warm-up runs
    for _ in range(10):
        probs, cache = forward(X_batch, params)
        backward(X_batch, y_batch, params, cache)

    # -----------------------------------------------------
    # Forward timing
    # -----------------------------------------------------
    forward_times = []

    for _ in range(runs):
        start = time.perf_counter()

        probs, cache = forward(
            X_batch,
            params
        )

        end = time.perf_counter()

        forward_times.append(
            (end - start) * 1000
        )

    # -----------------------------------------------------
    # Prepare cache for backward timing
    # -----------------------------------------------------
    probs, cache = forward(
        X_batch,
        params
    )

    # -----------------------------------------------------
    # Backward timing
    # -----------------------------------------------------
    backward_times = []

    for _ in range(runs):
        start = time.perf_counter()

        gradients = backward(
            X_batch,
            y_batch,
            params,
            cache
        )

        end = time.perf_counter()

        backward_times.append(
            (end - start) * 1000
        )

    forward_median = np.median(forward_times)
    backward_median = np.median(backward_times)

    return forward_median, backward_median


print("\n========================================")
print("TASK 7 - TIMING")
print("Median of 100 runs")
print("========================================")

for width in [16, 64]:

    forward_time, backward_time = measure_timing(
        width,
        runs=100
    )

    print(f"\nHidden width W = {width}")
    print(
        f"Forward median:  {forward_time:.6f} ms"
    )
    print(
        f"Backward median: {backward_time:.6f} ms"
    )

    if backward_time > forward_time:
        print("Backward pass is slower.")
    else:
        print("Forward pass is slower.")