"""Perceptron and logistic-regression review using only NumPy and Matplotlib."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def make_separable_data(n_samples=160, seed=7):
    """Make a 2-D set separated by ``y = sign(w·x+b)`` with a positive margin."""
    rng = np.random.default_rng(seed)
    direction = np.asarray([1.0, 0.6])
    direction /= np.linalg.norm(direction)
    tangent = np.asarray([-direction[1], direction[0]])
    labels = rng.choice(np.asarray([-1, 1]), size=n_samples)
    margins = rng.uniform(0.35, 1.4, size=n_samples)
    positions = rng.uniform(-2.0, 2.0, size=n_samples)
    features = labels[:, None] * margins[:, None] * direction + positions[:, None] * tangent
    return features, labels


def predict_perceptron(features, weight, bias):
    """Predict ``sign(w·x+b)``; an exact zero is assigned to the positive class."""
    return np.where(features @ weight + bias >= 0, 1, -1)


def train_perceptron(features, labels, learning_rate=0.1, max_epochs=1000, seed=0):
    """Apply ``w←w+ηyx`` and ``b←b+ηy`` whenever ``y(w·x+b)≤0``."""
    rng = np.random.default_rng(seed)
    weight = np.zeros(features.shape[1], dtype=np.float64)
    bias = 0.0
    mistakes_per_epoch = []
    for _ in range(max_epochs):
        mistakes = 0
        for index in rng.permutation(len(features)):
            label = labels[index]
            if label * (features[index] @ weight + bias) <= 0:
                weight += learning_rate * label * features[index]
                bias += learning_rate * label
                mistakes += 1
        mistakes_per_epoch.append(mistakes)
        if mistakes == 0:
            break
    return weight, bias, mistakes_per_epoch


def train_logistic_regression(features, labels, learning_rate=0.2, steps=800):
    """Minimize mean binary cross-entropy using ``∂BCE/∂z = sigmoid(z)-y``."""
    targets = (labels > 0).astype(np.float64)
    weight = np.zeros(features.shape[1], dtype=np.float64)
    bias = 0.0
    for _ in range(steps):
        logits = features @ weight + bias
        probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
        # For z=Xw+b, this is the Linear-layer backward: dW=Xᵀdz/N, db=Σdz/N.
        dlogits = probability - targets
        weight -= learning_rate * (features.T @ dlogits) / len(features)
        bias -= learning_rate * dlogits.mean()
    return weight, bias


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "runs" / "warmup"))
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args(argv)

    features, labels = make_separable_data()
    weight, bias, mistakes = train_perceptron(features, labels)
    logistic_weight, logistic_bias = train_logistic_regression(features, labels)
    x_values = np.linspace(features[:, 0].min() - 0.5, features[:, 0].max() + 0.5, 200)
    perceptron_y = -(weight[0] * x_values + bias) / weight[1]
    logistic_y = -(logistic_weight[0] * x_values + logistic_bias) / logistic_weight[1]

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for label, color, name in ((-1, "tab:blue", "−1"), (1, "tab:orange", "+1")):
        selected = labels == label
        axes[0].scatter(features[selected, 0], features[selected, 1], s=24, color=color, label=name)
    axes[0].plot(x_values, perceptron_y, color="black", label="perceptron boundary")
    axes[0].plot(x_values, logistic_y, color="green", linestyle="--", label="logistic boundary")
    axes[0].set(xlabel="$x_1$", ylabel="$x_2$", title="Linear decision boundaries")
    axes[0].legend()
    axes[1].plot(np.arange(1, len(mistakes) + 1), mistakes, marker="o")
    axes[1].set(xlabel="Epoch", ylabel="Mistakes", title="Perceptron convergence")
    axes[1].grid(alpha=0.3)
    figure.tight_layout()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "perceptron.png"
    figure.savefig(path, dpi=140)
    print(f"perceptron mistakes={mistakes[-1]} after {len(mistakes)} epochs; plot={path}")
    if args.show:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
