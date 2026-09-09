import numpy as np


def compute_specialization(samples: dict, stats: list, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    u_by_task = {}
    for stat in stats:
        mu_e = np.asarray(samples[f"mu__{stat.name}"])
        beta_e = np.asarray(samples[f"beta__{stat.name}"])
        omega_e = np.asarray(samples[f"omega__{stat.name}"])
        theta_e = stat.theta_samples

        r_e = stat.y_hat[None, :] - mu_e[:, None] + beta_e[:, None] * theta_e

        lam, Q = np.linalg.eigh(stat.V)
        lam = np.clip(lam, 0.0, None)

        w2 = omega_e[:, None] ** 2
        denom = lam[None, :] + w2
        r_rot = r_e @ Q
        mean_rot = w2 * r_rot / denom
        sd_rot = np.sqrt(w2 * lam[None, :] / denom)
        z = rng.standard_normal(mean_rot.shape)
        u_by_task[stat.name] = (mean_rot + sd_rot * z) @ Q.T
    return u_by_task
