import logging as log

import numpy as np

from blam_analysis.data import build_stats, compute_prior_scale
from blam_analysis.inference import fit_posterior
from blam_analysis.result import BLAMResult
from blam_analysis.specialization import compute_specialization
from blam_analysis.summarize import summarize

__all__ = ["fit", "BLAMResult"]

log.basicConfig(level=log.INFO, format="%(asctime)s %(levelname)s %(message)s")


def fit(
    df,
    num_warmup: int = 1000,
    num_samples: int = 2000,
    num_chains: int = 4,
    seed: int = 0,
    backend: str = "auto",
    n_workers: int = None,
    specialization_seed: int = 0,
    min_models_per_task: int = 2,
) -> BLAMResult:
    """
    Fit the BLAM model to predictions and compute the derived summaries.

    Args:
        df: long-format DataFrame with columns 'model', 'task', 'sample_id', 'y_true', 'y_pred'
        num_warmup, num_samples, num_chains, seed: NUTS sampling parameters.
        backend: "auto" (default), "cpu", or "gpu". 
            "auto" detects via JAX's own default backend. 
            CPU fans chains out across OS processes
            GPU runs all chains vectorized in one process.
        n_workers: CPU backend only, number of worker processes (default: min(num_chains, os.cpu_count())). Ignored on GPU.
        specialization_seed: seed for the per-task specialization (u_ie) posterior-conditional draw.
        min_models_per_task: tasks with fewer models than this are dropped

    Returns:
        A BLAMResult with theta_summary (model ranking), specialization_summary
        (per-task strengths/weaknesses), task_summary (difficulty/discrimination/
        noise diagnostics), and comparison helpers.
    """
    stats = build_stats(df, min_models_per_task=min_models_per_task)
    if not stats:
        raise ValueError(f"No task had >= {min_models_per_task} models -- nothing to fit")

    global_models = sorted({m for stat in stats for m in stat.models})
    log.info(f"Fitting {len(global_models)} models across {len(stats)} tasks")

    prior_log_s = float(np.log(compute_prior_scale(stats) / np.sqrt(2)))
    samples, grouped_samples = fit_posterior(
        global_models, stats, prior_log_s, num_warmup, num_samples, num_chains, seed, backend, n_workers
    )

    model_index = {name: i for i, name in enumerate(global_models)}
    theta = np.asarray(samples["theta"])
    for stat in stats:
        idx = [model_index[m] for m in stat.models]
        stat.theta_samples = theta[:, idx]

    u_by_task = compute_specialization(samples, stats, specialization_seed)
    theta_summary, specialization_summary, task_summary = summarize(global_models, samples, stats, u_by_task)

    return BLAMResult(
        samples=samples,
        grouped_samples=grouped_samples,
        stats=stats,
        u_by_task=u_by_task,
        global_models=global_models,
        theta_summary=theta_summary,
        specialization_summary=specialization_summary,
        task_summary=task_summary,
    )
