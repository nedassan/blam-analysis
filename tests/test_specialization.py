import numpy as np

from blam_analysis import specialization


def _random_psd(n_e, rng, scale=1.0):
    M = rng.normal(size=(n_e, n_e))
    return scale * (M @ M.T) / n_e


class _FakeStat:
    def __init__(self, name, y_hat, V):
        self.name = name
        self.models = [f"model{i}" for i in range(len(y_hat))]
        self.y_hat = y_hat
        self.V = V


def _samples_and_stat(n_e, S, omega, V, r):
    stat = _FakeStat("e", np.zeros(n_e), V)
    stat.theta_samples = r
    samples = {"mu__e": np.zeros(S), "beta__e": np.ones(S), "omega__e": np.full(S, omega)}
    return samples, stat


def test_marginal_consistency():
    rng = np.random.default_rng(0)
    n_e, S, omega = 5, 200_000, 0.7
    V = _random_psd(n_e, rng, scale=0.3)
    r = rng.multivariate_normal(np.zeros(n_e), omega**2 * np.eye(n_e) + V, size=S)
    samples, stat = _samples_and_stat(n_e, S, omega, V, r)

    u = specialization.compute_specialization(samples, [stat], seed=2)["e"]
    assert np.allclose(u.var(axis=0), omega**2, rtol=0.05)


def test_limit_zero_noise():
    n_e, S, omega = 4, 1000, 0.5
    V = np.eye(n_e) * 1e-12
    rng = np.random.default_rng(3)
    r = rng.normal(size=(S, n_e))
    samples, stat = _samples_and_stat(n_e, S, omega, V, r)

    u = specialization.compute_specialization(samples, [stat], seed=4)["e"]
    assert np.allclose(u, r, atol=1e-4)


def test_limit_large_noise_reverts_to_prior():
    n_e, S, omega = 4, 200_000, 0.3
    V = np.eye(n_e) * 1e6
    rng = np.random.default_rng(5)
    r = rng.normal(size=(S, n_e))
    samples, stat = _samples_and_stat(n_e, S, omega, V, r)

    u = specialization.compute_specialization(samples, [stat], seed=6)["e"]
    assert np.allclose(u.mean(axis=0), 0.0, atol=0.02)
    assert np.allclose(u.var(axis=0), omega**2, rtol=0.05)


def test_regression_guard_mean_matches_old_plugin(monkeypatch):
    class _ZeroRNG:
        def standard_normal(self, shape):
            return np.zeros(shape)

    n_e, S, omega = 6, 50, 0.4
    rng = np.random.default_rng(7)
    V = _random_psd(n_e, rng, scale=0.4)
    r = rng.normal(size=(S, n_e))
    samples, stat = _samples_and_stat(n_e, S, omega, V, r)

    monkeypatch.setattr(specialization.np.random, "default_rng", lambda seed: _ZeroRNG())
    u = specialization.compute_specialization(samples, [stat], seed=0)["e"]

    Sigma_e = omega**2 * np.eye(n_e) + V
    g_e = np.linalg.solve(Sigma_e, r[:, :, None])[:, :, 0]
    u_old = omega**2 * g_e
    assert np.allclose(u, u_old, atol=1e-8)
