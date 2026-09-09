import os

import click
import pandas as pd

import blam_analysis
from blam_analysis.viz.common import load_display_names


def _read_predictions(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


@click.command()
@click.option(
    "--input", "input_path", required=True, type=click.Path(exists=True, dir_okay=False),
    help="CSV or Parquet file with columns: model, task, sample_id, y_true, y_pred",
)
@click.option("--output", default="blam_output", show_default=True, help="Directory to write posterior.joblib + summary CSVs")
@click.option("--num-warmup", default=1000, show_default=True, type=int)
@click.option("--num-samples", default=2000, show_default=True, type=int)
@click.option("--num-chains", default=4, show_default=True, type=int)
@click.option("--seed", default=0, show_default=True, type=int)
@click.option("--backend", default="auto", show_default=True, type=click.Choice(["auto", "cpu", "gpu"]))
@click.option(
    "--n-workers", default=None, type=int,
    help="CPU backend only: worker processes (default: min(num_chains, cpu_count))",
)
@click.option("--specialization-seed", default=0, show_default=True, type=int)
@click.option("--min-models-per-task", default=2, show_default=True, type=int)
@click.option(
    "--display-names", default=None, type=click.Path(exists=True, dir_okay=False),
    help="Path to a JSON {raw_model_name: display_name} map",
)
@click.option("--plot-theta", is_flag=True, help="Write a static bar chart (PNG) of the theta ranking")
@click.option("--dashboard", is_flag=True, help="Write an interactive HTML dashboard")
@click.option("--compare-baseline", default=None, help="Model name to compare every other model against")
@click.option("--alpha", default=0.05, show_default=True, type=float, help="Significance threshold for --compare-baseline's p-value")
def main(
    input_path, output, num_warmup, num_samples, num_chains, seed, backend, n_workers,
    specialization_seed, min_models_per_task, display_names, plot_theta, dashboard, compare_baseline, alpha,
):
    df = _read_predictions(input_path)
    result = blam_analysis.fit(
        df,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        seed=seed,
        backend=backend,
        n_workers=n_workers,
        specialization_seed=specialization_seed,
        min_models_per_task=min_models_per_task,
    )
    result.save(output)
    click.echo("\nLatent ability ranking (theta, higher = better):")
    click.echo(result.theta_summary.to_string(index=False))

    display_names_map = load_display_names(display_names)
    if plot_theta or dashboard:
        from blam_analysis import viz

        if plot_theta:
            viz.plot_theta(result.theta_summary, os.path.join(output, "theta_ranking.png"), display_names_map)
        if dashboard:
            mae_df = result.mae_summary()
            viz.write_dashboard_html(
                result.theta_summary, result.specialization_summary, result.task_summary,
                os.path.join(output, "dashboard.html"), mae_df, display_names_map,
            )

    if compare_baseline:
        comparison_df = result.compare(compare_baseline, seed=seed)
        comparison_df.to_csv(os.path.join(output, "comparison.csv"), index=False)
        click.echo(f"\nP(model beats {compare_baseline}):")
        click.echo(comparison_df.to_string(index=False))


if __name__ == "__main__":
    main()
