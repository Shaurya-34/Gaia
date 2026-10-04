"""PREREG section 3 unit test for the vendored CST: a Gaussian blob in uniform noise."""
import numpy as np

from ocelot_cst import cluster_significance_test


def blob_in_noise(rng, n_noise=5000, n_blob=60):
    noise = rng.uniform(0, 1, (n_noise, 5))
    blob = rng.normal(0.5, 0.01, (n_blob, 5))
    X = np.vstack([noise, blob])
    labels = np.r_[np.full(n_noise, -1), np.zeros(n_blob, dtype=int)]
    return X, labels


def test_blob_is_significant():
    X, labels = blob_in_noise(np.random.default_rng(0))
    sig, _ = cluster_significance_test(X, labels, min_samples=10)
    assert sig[0] > 5


def test_random_noise_group_is_not_significant():
    rng = np.random.default_rng(1)
    X, labels = blob_in_noise(rng)
    labels[:] = -1
    labels[rng.choice(5000, 60, replace=False)] = 0  # 60 random noise points called a "cluster"
    sig, _ = cluster_significance_test(X, labels, min_samples=10)
    assert sig[0] < 2


def test_deterministic():
    X, labels = blob_in_noise(np.random.default_rng(2))
    assert cluster_significance_test(X, labels)[0] == cluster_significance_test(X, labels)[0]
