import logging as log

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("model", "task", "sample_id", "y_true", "y_pred")


def validate_predictions(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"predictions DataFrame is missing required column(s): {missing}")
    dupes = df.duplicated(subset=["model", "task", "sample_id"])
    if dupes.any():
        example = df.loc[dupes, ["model", "task", "sample_id"]].iloc[0]
        raise ValueError(
            f"duplicate (model, task, sample_id) row, e.g. "
            f"model={example['model']!r} task={example['task']!r} sample_id={example['sample_id']!r}"
        )


def ensure_psd(V: np.ndarray, floor: float = 1e-10) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(V)
    if eigvals.min() >= floor:
        return V
    eigvals = np.clip(eigvals, floor, None)
    V_psd = (eigvecs * eigvals) @ eigvecs.T
    return (V_psd + V_psd.T) / 2


def compute_sampling_stats(A: pd.DataFrame):
    X = A.to_numpy()
    n_samples, n_models = X.shape
    y_hat = np.nanmean(X, axis=0)
    V = np.empty((n_models, n_models))
    n_pairs = np.empty((n_models, n_models), dtype=int)
    for i in range(n_models):
        xi = X[:, i]
        for j in range(i, n_models):
            xj = X[:, j]
            mask = ~np.isnan(xi) & ~np.isnan(xj)
            n_ij = int(mask.sum())
            if n_ij < 2:
                raise ValueError(
                    f"models '{A.columns[i]}' and '{A.columns[j]}' share fewer than 2 "
                    "samples -- can't estimate their sampling covariance"
                )
            cov = np.sum((xi[mask] - y_hat[i]) * (xj[mask] - y_hat[j])) / (n_ij - 1)
            V[i, j] = V[j, i] = cov / n_ij
            n_pairs[i, j] = n_pairs[j, i] = n_ij
    return y_hat, ensure_psd(V), n_pairs


class TaskStat:
    def __init__(self, name, models, y_hat, V, n_samples):
        self.name = name
        self.models = models
        self.y_hat = y_hat
        self.V = V
        self.n_samples = n_samples


def build_stats(df: pd.DataFrame, min_models_per_task: int = 2) -> list:
    validate_predictions(df)
    stats = []
    for task, task_df in df.groupby("task", sort=True):
        error = (task_df["y_pred"].to_numpy(dtype=float) - task_df["y_true"].to_numpy(dtype=float))
        long = pd.DataFrame({
            "sample_id": task_df["sample_id"].to_numpy(),
            "model": task_df["model"].to_numpy(),
            "abs_error": np.abs(error),
        })
        A = long.pivot(index="sample_id", columns="model", values="abs_error")
        if A.shape[1] < min_models_per_task:
            log.warning(f"'{task}': only {A.shape[1]} model(s) -- skipping task (need >= {min_models_per_task})")
            continue
        y_hat, V, n_pairs = compute_sampling_stats(A)
        stats.append(TaskStat(str(task), list(A.columns), y_hat, V, A.shape[0]))
    return stats


EPSILON_0 = 1e-6


def compute_prior_scale(stats: list) -> float:
    s_tilde = []
    for stat in stats:
        s_e2 = np.var(stat.y_hat, ddof=1)
        diag = np.diag(stat.V)
        off_diag_sum = stat.V.sum() - diag.sum()
        n = len(stat.models)
        v_bar = diag.mean()
        c_bar = off_diag_sum / (n * n - n)
        s_tilde.append(np.sqrt(max(s_e2 - (v_bar - c_bar), EPSILON_0)))
    return float(np.median(s_tilde))
