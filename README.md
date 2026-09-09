# blam-analysis

Bayesian Latent Ability Model (BLAM): rank a set of models across a set of
tasks by a shared latent "ability", from each model's raw predictions and
ground truth on each task.

Given a model's per-task predictions, `blam_analysis` computes mean absolute error
(MAE) itself and fits a hierarchical Bayesian model (via NUTS/NumPyro) that
decomposes each model's per-task log-MAE into a task difficulty, a task
discrimination, a model's overall latent ability, and a per-(model, task)
specialization term. Output is a ranking of models by latent ability, with
uncertainty, plus per-task diagnostics and head-to-head comparison
probabilities.

## Install

```
uv sync
uv sync --extra viz  # also installs matplotlib, for plot_theta/dashboard
```

GPU support: install a CUDA-matched `jax` wheel yourself (this package
doesn't pin a CUDA extra, since the right wheel depends on your CUDA
version) -- see [JAX's install instructions](https://jax.readthedocs.io/en/latest/installation.html).
With a GPU visible, `blam_analysis.fit(..., backend="auto")` runs all chains
vectorized on it automatically; without one, it fans chains out across CPU
processes.

## Usage

```python
import pandas as pd
import blam_analysis

# One row per (model, task, sample_id, y_true, y_pred). sample_id identifies
# the example -- used to compute the sampling covariance between models that
# share test examples on a task.
df = pd.DataFrame({
    "model": [...],
    "task": [...],
    "sample_id": [...],
    "y_true": [...],
    "y_pred": [...],
})

result = blam_analysis.fit(df, num_warmup=1000, num_samples=2000, num_chains=4)

result.theta_summary          # model, theta_mean, theta_std -- ranked, higher = better
result.specialization_summary # task, model, u_mean, u_std
result.task_summary           # task, mu/beta/omega mean+std, n_samples, n_models

result.compare("some_baseline_model")          # P(model beats baseline) on a new/unseen task
result.compare_per_task("some_baseline_model")  # P(model beats baseline) on each task actually observed

result.save("out/")            # posterior.joblib + 3 summary CSVs
blam_analysis.BLAMResult.load("out/")
```

Optional plots/dashboard (`blam-analysis[viz]`):

```python
from blam_analysis import viz

viz.plot_theta(result.theta_summary, "theta_ranking.png")
viz.write_dashboard_html(result.theta_summary, result.specialization_summary, result.task_summary, "dashboard.html")

comparison_df = result.compare("some_baseline_model")
viz.plot_comparison(comparison_df, "comparison.png", baseline="some_baseline_model", alpha=0.05)
```

## CLI

```
blam-analysis --input predictions.csv --output out/ --dashboard --plot-theta --compare-baseline some_baseline_model
```

`--input` is a CSV or Parquet file with columns `model, task, sample_id,
y_true, y_pred`.

## Backend / parallelism

- `backend="auto"` (default) detects via JAX's own default backend.
- CPU: chains fan out across separate OS processes (`ProcessPoolExecutor`,
  spawn start method) so they don't contend for the same cores the way
  NumPyro's own in-process chain parallelism can.
- GPU: all chains run vectorized (`chain_method="vectorized"`, batched via
  `vmap`) in a single process, since one GPU doing one batched XLA call is
  more efficient than multiple processes contending for the same device.
  Multi-GPU is not supported yet.
