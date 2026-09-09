import numpy as np
import pandas as pd
import pytest

import blam_analysis


def _synthetic_predictions(n_models=4, n_tasks=3, n_samples=25, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_tasks):
        y_true = rng.normal(size=n_samples)
        for m in range(n_models):
            noise = rng.normal(scale=0.5 + 0.1 * m, size=n_samples)
            y_pred = y_true + noise
            for s in range(n_samples):
                rows.append({
                    "model": f"model{m}",
                    "task": f"task{t}",
                    "sample_id": s,
                    "y_true": y_true[s],
                    "y_pred": y_pred[s],
                })
    return pd.DataFrame(rows)


def test_fit_cpu_multiprocess_end_to_end():
    df = _synthetic_predictions()
    result = blam_analysis.fit(
        df, num_warmup=50, num_samples=50, num_chains=2, seed=0, backend="cpu", n_workers=2,
    )

    assert set(result.theta_summary["model"]) == {f"model{i}" for i in range(4)}
    assert np.isclose(result.theta_summary["theta_mean"].sum(), 0.0, atol=0.5)
    assert set(result.task_summary["task"]) == {f"task{i}" for i in range(3)}
    assert not result.specialization_summary.empty


def test_fit_raises_when_no_task_has_enough_models():
    df = pd.DataFrame([
        {"model": "m1", "task": "t1", "sample_id": "s1", "y_true": 1.0, "y_pred": 1.1},
        {"model": "m1", "task": "t1", "sample_id": "s2", "y_true": 2.0, "y_pred": 2.1},
    ])
    with pytest.raises(ValueError, match="nothing to fit"):
        blam_analysis.fit(df)


@pytest.mark.skipif(
    __import__("jax").default_backend() != "gpu", reason="requires a GPU-backed JAX install"
)
def test_fit_gpu_vectorized_end_to_end():
    df = _synthetic_predictions()
    result = blam_analysis.fit(df, num_warmup=50, num_samples=50, num_chains=2, seed=0, backend="gpu")
    assert set(result.theta_summary["model"]) == {f"model{i}" for i in range(4)}
