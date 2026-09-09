import numpy as np
import pandas as pd

import blam_analysis


def make_synthetic_predictions(n_models=5, n_tasks=4, n_samples=200, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_tasks):
        y_true = rng.normal(size=n_samples)
        for m in range(n_models):
            noise = rng.normal(scale=0.3 + 0.15 * m, size=n_samples)
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


def main():
    df = make_synthetic_predictions()
    result = blam_analysis.fit(df, num_warmup=1000, num_samples=2000, num_chains=4, seed=0)

    print("\nLatent ability ranking (theta, higher = better):")
    print(result.theta_summary.to_string(index=False))

    print("\nP(model beats model4):")
    print(result.compare("model4").to_string(index=False))

    result.save("out/")
    print("\nWrote out/posterior.joblib + summary CSVs")


if __name__ == "__main__":
    main()
