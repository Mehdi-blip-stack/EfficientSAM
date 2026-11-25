"""PCA by SVD and by hand-written covariance power iteration."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def pca_svd(data):
    """Center ``X`` and use ``X=UΣVᵀ``; columns of ``V`` are principal directions."""
    data = np.asarray(data, dtype=np.float64)
    mean = data.mean(axis=0)
    centered = data - mean
    _left, singular_values, right = np.linalg.svd(centered, full_matrices=False)
    eigenvalues = singular_values ** 2 / max(len(data) - 1, 1)
    return mean, right.T, eigenvalues


def pca_power_iteration(data, n_components, seed=0, tolerance=1e-12, max_iterations=3000):
    """Find covariance eigenvectors by iteration, then deflate ``C←C−λvvᵀ``."""
    data = np.asarray(data, dtype=np.float64)
    mean = data.mean(axis=0)
    centered = data - mean
    covariance = centered.T @ centered / max(len(data) - 1, 1)
    residual = covariance.copy()
    rng = np.random.default_rng(seed)
    components, eigenvalues = [], []
    for _ in range(n_components):
        vector = rng.normal(size=data.shape[1])
        vector /= np.linalg.norm(vector)
        for _ in range(max_iterations):
            updated = residual @ vector
            norm = np.linalg.norm(updated)
            if norm < tolerance:
                vector = np.zeros_like(vector)
                break
            updated /= norm
            difference = min(np.linalg.norm(updated - vector), np.linalg.norm(updated + vector))
            vector = updated
            if difference < tolerance:
                break
        eigenvalue = max(float(vector @ residual @ vector), 0.0)
        components.append(vector)
        eigenvalues.append(eigenvalue)
        residual -= eigenvalue * np.outer(vector, vector)
        residual = 0.5 * (residual + residual.T)
    return mean, np.column_stack(components), np.asarray(eigenvalues)


def reconstruct(data, mean, components, n_components):
    """Project onto the first ``k`` directions and map the scores back: ``μ+(X−μ)VₖVₖᵀ``."""
    centered = np.asarray(data, dtype=np.float64) - mean
    basis = components[:, :n_components]
    return mean + (centered @ basis) @ basis.T


def reconstruction_error(data, mean, components, n_components):
    """Return mean squared pixel/feature error after rank-``k`` reconstruction."""
    residual = np.asarray(data, dtype=np.float64) - reconstruct(data, mean, components, n_components)
    return float(np.mean(residual ** 2))


def make_correlated_data(n_samples=600, seed=17):
    """Make five correlated features from two strong latent factors plus small noise."""
    rng = np.random.default_rng(seed)
    factors = rng.normal(size=(n_samples, 2)) * np.asarray([3.5, 1.5])
    mixing = np.asarray([
        [1.0, 0.1], [0.8, -0.2], [0.7, 0.5], [-0.3, 1.0], [0.2, 0.7],
    ])
    return factors @ mixing.T + rng.normal(scale=0.12, size=(n_samples, 5))


def plot_pets(cache_path, output):
    """Show leading eigenpets and rank-5/20/50 reconstructions if a 64px cache exists."""
    if not cache_path.exists():
        print(f"Skipping Pets PCA; cache not found: {cache_path}")
        return
    with np.load(cache_path) as cache:
        images = cache["images"][:256].astype(np.float32)
    small = images.reshape(len(images), 3, 32, 2, 32, 2).mean(axis=(3, 5))
    grayscale = 0.299 * small[:, 0] + 0.587 * small[:, 1] + 0.114 * small[:, 2]
    data = grayscale.reshape(len(grayscale), -1)
    mean, components, _eigenvalues = pca_svd(data)

    figure, axes = plt.subplots(4, 4, figsize=(7, 7))
    for index, axis in enumerate(axes.flat):
        eigenpet = components[:, index].reshape(32, 32)
        low, high = eigenpet.min(), eigenpet.max()
        axis.imshow((eigenpet - low) / max(high - low, 1e-12), cmap="gray")
        axis.set_title(f"PC {index + 1}")
        axis.axis("off")
    figure.tight_layout()
    eigen_path = output / "eigenpets.png"
    figure.savefig(eigen_path, dpi=140)
    plt.close(figure)

    count = min(5, len(data))
    figure, axes = plt.subplots(count, 4, figsize=(9, 2.1 * count), squeeze=False)
    for index in range(count):
        axes[index, 0].imshow(data[index].reshape(32, 32), cmap="gray", vmin=0, vmax=255)
        axes[index, 0].set_title("Original" if index == 0 else "")
        for column, components_count in enumerate((5, 20, 50), start=1):
            reconstruction = reconstruct(data[index:index + 1], mean, components, components_count)[0]
            axes[index, column].imshow(reconstruction.reshape(32, 32), cmap="gray", vmin=0, vmax=255)
            axes[index, column].set_title(f"k={components_count}" if index == 0 else "")
        for axis in axes[index]:
            axis.axis("off")
    figure.tight_layout()
    reconstruction_path = output / "pets_reconstructions.png"
    figure.savefig(reconstruction_path, dpi=140)
    plt.close(figure)
    print(f"eigenpets={eigen_path}\nPets reconstructions={reconstruction_path}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "runs" / "warmup"))
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--pets", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)

    data = make_correlated_data()
    mean, components, eigenvalues = pca_svd(data)
    _power_mean, power_components, power_eigenvalues = pca_power_iteration(
        data, n_components=data.shape[1], seed=5
    )
    ratios = eigenvalues / eigenvalues.sum()
    errors = [reconstruction_error(data, mean, components, k)
              for k in range(1, data.shape[1] + 1)]

    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].bar(np.arange(1, len(ratios) + 1), ratios, label="individual")
    axes[0].plot(np.arange(1, len(ratios) + 1), np.cumsum(ratios), "o-", label="cumulative")
    axes[0].set(xlabel="Principal component", ylabel="Explained variance ratio",
                title="Variance captured")
    axes[0].legend()
    sample = data[0]
    axes[1].plot(np.arange(1, len(sample) + 1), sample, "ko-", label="original")
    for k, style in ((1, "--"), (2, "-."), (5, ":")):
        estimate = reconstruct(sample[None], mean, components, k)[0]
        axes[1].plot(np.arange(1, len(sample) + 1), estimate, style, marker=".", label=f"k={k}")
    axes[1].set(xlabel="Feature", ylabel="Value", title="One sample reconstruction")
    axes[1].legend()
    axes[2].plot(np.arange(1, len(errors) + 1), errors, "o-")
    axes[2].set(xlabel="Components k", ylabel="Mean squared error", title="Reconstruction error")
    axes[2].grid(alpha=0.3)
    figure.tight_layout()
    path = output / "pca_svd.png"
    figure.savefig(path, dpi=140)
    print(f"PC1 SVD/power absolute cosine={abs(components[:, 0] @ power_components[:, 0]):.6f}; "
          f"plot={path}")
    if args.show:
        plt.show()
    plt.close(figure)
    if args.pets:
        plot_pets(Path(__file__).resolve().parents[1] / "data" / "cache" / "pets_train_64.npz", output)


if __name__ == "__main__":
    main()
