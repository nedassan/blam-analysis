import numpy as np
import pandas as pd


def summarize(global_models: list, samples: dict, stats: list, u_by_task: dict):
    theta = np.asarray(samples["theta"])
    theta_summary = pd.DataFrame({
        "model": global_models,
        "theta_mean": theta.mean(axis=0),
        "theta_std": theta.std(axis=0),
    }).sort_values("theta_mean", ascending=False).reset_index(drop=True)

    rows = []
    for stat in stats:
        u_e = u_by_task[stat.name]
        for j, model in enumerate(stat.models):
            rows.append({
                "task": stat.name,
                "model": model,
                "u_mean": u_e[:, j].mean(),
                "u_std": u_e[:, j].std(),
            })
    specialization_summary = pd.DataFrame(rows)

    task_rows = []
    for stat in stats:
        mu = np.asarray(samples[f"mu__{stat.name}"])
        beta = np.asarray(samples[f"beta__{stat.name}"])
        omega = np.asarray(samples[f"omega__{stat.name}"])
        task_rows.append({
            "task": stat.name,
            "mu_mean": mu.mean(),
            "mu_std": mu.std(),
            "beta_mean": beta.mean(),
            "beta_std": beta.std(),
            "omega_mean": omega.mean(),
            "omega_std": omega.std(),
            "n_samples": stat.n_samples,
            "n_models": len(stat.models),
        })
    task_summary = pd.DataFrame(task_rows)

    return theta_summary, specialization_summary, task_summary
