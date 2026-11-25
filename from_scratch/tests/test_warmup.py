import numpy as np

from warmup.knn import knn_predict, make_blobs, squared_distance_matrix, stratified_split
from warmup.pca_svd import (
    make_correlated_data, pca_power_iteration, pca_svd, reconstruction_error,
)
from warmup.perceptron import make_separable_data, predict_perceptron, train_perceptron


def test_perceptron_learns_separable_data_without_mistakes():
    features, labels = make_separable_data(seed=24)
    weight, bias, mistakes = train_perceptron(features, labels, seed=3)
    assert mistakes[-1] == 0
    np.testing.assert_array_equal(predict_perceptron(features, weight, bias), labels)


def test_knn_distances_match_brute_force_and_training_one_nn_is_exact():
    features = np.asarray([[0.0, 1.0], [2.0, -1.0], [-3.0, 2.0]])
    queries = np.asarray([[1.0, 2.0], [-1.0, 0.5]])
    expected = np.sum((queries[:, None, :] - features[None, :, :]) ** 2, axis=2)
    np.testing.assert_allclose(squared_distance_matrix(queries, features), expected)

    train_x, train_y, _, _ = stratified_split(*make_blobs(seed=9), seed=4)
    np.testing.assert_array_equal(knn_predict(train_x, train_y, train_x, k=1), train_y)


def test_knn_vote_ties_choose_smallest_label():
    train_x = np.asarray([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    train_y = np.asarray([2, 1, 2, 1])
    assert knn_predict(train_x, train_y, np.asarray([[0.0, 0.0]]), k=4)[0] == 1


def test_power_iteration_matches_svd_and_reconstruction_error_falls():
    data = make_correlated_data(n_samples=300, seed=8)
    mean, svd_components, svd_values = pca_svd(data)
    power_mean, power_components, power_values = pca_power_iteration(data, 5, seed=2)
    np.testing.assert_allclose(power_mean, mean)
    np.testing.assert_allclose(power_values, svd_values, rtol=1e-4, atol=1e-8)
    cosines = np.abs(np.sum(svd_components * power_components, axis=0))
    assert np.all(cosines > 0.999)
    errors = [reconstruction_error(data, mean, svd_components, k) for k in range(1, 6)]
    assert np.all(np.diff(errors) <= 1e-12)
