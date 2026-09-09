import numpy as np
import pandas as pd
from scipy.special import ndtr


def compute_mae_summary(result) -> pd.DataFrame:
    rows = []
    for stat in result.stats:
        se = np.sqrt(np.diag(stat.V))
        for j, model in enumerate(stat.models):
            rows.append({
                "task": stat.name,
                "model": model,
                "mae_mean": float(stat.y_hat[j]),
                "mae_se": float(se[j]),
            })
    return pd.DataFrame(rows)


def compare_to_baseline(result, baseline: str, seed: int = 0) -> pd.DataFrame:
    global_models = result.global_models
    if baseline not in global_models:
        raise ValueError(f"baseline '{baseline}' not found among fitted models: {global_models}")
    samples = result.samples
    theta = np.asarray(samples["theta"])
    theta_baseline = theta[:, global_models.index(baseline)]

    rng = np.random.default_rng(seed)
    sigma_beta = np.asarray(samples["sigma_beta"])
    sigma_omega = np.asarray(samples["sigma_omega"])
    beta_new = np.abs(rng.normal(0.0, sigma_beta))
    omega_new = np.abs(rng.normal(0.0, sigma_omega))

    rows = []
    for i, model in enumerate(global_models):
        if model == baseline:
            continue
        delta = theta[:, i] - theta_baseline
        prob_draws = ndtr(beta_new * delta / (np.sqrt(2) * omega_new))
        prob_beat = float(prob_draws.mean())
        rows.append({
            "model": model,
            "prob_beat_baseline": prob_beat,
            "p_value": 2 * min(prob_beat, 1 - prob_beat),
            "delta_mean": float(delta.mean()),
        })
    return pd.DataFrame(rows).sort_values("prob_beat_baseline").reset_index(drop=True)


def compare_to_baseline_per_task(result, baseline: str) -> pd.DataFrame:
    global_models = result.global_models
    if baseline not in global_models:
        raise ValueError(f"baseline '{baseline}' not found among fitted models: {global_models}")
    samples = result.samples
    u_by_task = result.u_by_task

    rows = []
    for stat in result.stats:
        if baseline not in stat.models:
            continue
        beta_e = np.asarray(samples[f"beta__{stat.name}"])
        u_e = u_by_task[stat.name]
        baseline_col = stat.models.index(baseline)
        theta_baseline = stat.theta_samples[:, baseline_col]
        u_baseline = u_e[:, baseline_col]
        for col, model in enumerate(stat.models):
            if model == baseline:
                continue
            d = beta_e * (stat.theta_samples[:, col] - theta_baseline) + (u_baseline - u_e[:, col])
            prob_beat = float((d > 0).mean())
            rows.append({
                "task": stat.name,
                "model": model,
                "prob_beat_baseline": prob_beat,
                "p_value": 2 * min(prob_beat, 1 - prob_beat),
            })
    return pd.DataFrame(rows)
