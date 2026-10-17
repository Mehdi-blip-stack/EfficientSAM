"""A small K-nearest-neighbor classifier written with NumPy array operations."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def squared_distance_matrix(queries, references):
    """Compute ``||q-r||² = ||q||² + ||r||² − 2q·r`` without Python point loops."""
    queries = np.asarray(queries, dtype=np.float64)
    references = np.asarray(references, dtype=np.float64)
    if queries.ndim != 2 or references.ndim != 2 or queries.shape[1] != references.shape[1]:
        raise ValueError("queries and references must be 2-D arrays with matching feature widths")
    distances = (
        np.sum(queries * queries, axis=1, keepdims=True)
        + np.sum(references * references, axis=1)[None, :]
        - 2.0 * queries @ references.T
    )
    return np.maximum(distances, 0.0)


def knn_predict(train_features, train_labels, query_features, k=1):
    """Vote among the ``k`` nearest points; equal vote counts choose the smallest label."""
    if not 1 <= k <= len(train_features):
        raise ValueError("k must be between one and the number of training points")
    distances = squared_distance_matrix(query_features, train_features)
    nearest = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
    predictions = []
    for neighbors in nearest:
        labels, counts = np.unique(np.asarray(train_labels)[neighbors], return_counts=True)
        predictions.append(labels[counts == counts.max()][0])
    return np.asarray(predictions, dtype=np.asarray(train_labels).dtype)


def make_blobs(seed=11, points_per_class=120):
    """Create three labeled Gaussian blobs for a reproducible classification demo."""
    rng = np.random.default_rng(seed)
    centers = np.asarray([[-2.0, -1.5], [2.0, -1.0], [0.1, 2.2]])
    features = np.concatenate([
        rng.normal(loc=center, scale=0.65, size=(points_per_class, 2))
        for center in centers
    ])
    labels = np.repeat(np.arange(len(centers)), points_per_class)
    order = rng.permutation(len(features))
    return features[order], labels[order]


def stratified_split(features, labels, train_fraction=0.75, seed=3):
    """Split each class independently so both sets retain all three classes."""
    rng = np.random.default_rng(seed)
    train_indices, test_indices = [], []
    for label in np.unique(labels):
        indices = rng.permutation(np.flatnonzero(labels == label))
        boundary = int(len(indices) * train_fraction)
        train_indices.extend(indices[:boundary])
        test_indices.extend(indices[boundary:])
    return (features[train_indices], labels[train_indices],
            features[test_indices], labels[test_indices])


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "runs" / "warmup"))
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args(argv)

    features, labels = make_blobs()
    train_x, train_y, test_x, test_y = stratified_split(features, labels)
    candidates = np.arange(1, 26, 2)
    accuracies = np.asarray([
        np.mean(knn_predict(train_x, train_y, test_x, int(k)) == test_y)
        for k in candidates
    ])
    best_k = int(candidates[np.argmax(accuracies)])
    x_grid = np.linspace(features[:, 0].min() - 0.5, features[:, 0].max() + 0.5, 100)
    y_grid = np.linspace(features[:, 1].min() - 0.5, features[:, 1].max() + 0.5, 100)
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    regions = knn_predict(train_x, train_y, grid, best_k).reshape(xx.shape)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(candidates, accuracies, marker="o")
    axes[0].set(xticks=candidates, xlabel="k", ylabel="Test accuracy",
                title="KNN accuracy versus neighborhood size")
    axes[0].grid(alpha=0.3)
    axes[1].contourf(xx, yy, regions, alpha=0.25, levels=np.arange(-0.5, 3.5, 1), cmap="viridis")
    axes[1].scatter(train_x[:, 0], train_x[:, 1], c=train_y, s=16, cmap="viridis", label="train")
    axes[1].scatter(test_x[:, 0], test_x[:, 1], c=test_y, s=28, marker="x", cmap="viridis", label="test")
    axes[1].set(title=f"Decision regions (k={best_k}, accuracy={accuracies.max():.3f})",
                xlabel="$x_1$", ylabel="$x_2$")
    axes[1].legend()
    figure.tight_layout()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "knn.png"
    figure.savefig(path, dpi=140)
    print(f"best k={best_k}, test accuracy={accuracies.max():.3f}; plot={path}")
    if args.show:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
