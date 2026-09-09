import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from scipy.linalg import helmert


def build_model(global_models: list, stats: list, prior_log_s: float):
    n_models = len(global_models)
    model_index = {name: i for i, name in enumerate(global_models)}

    Q = jnp.array(helmert(n_models, full=False).T)

    def model():
        z = numpyro.sample("z", dist.Normal(0.0, 1.0).expand([n_models - 1]).to_event(1))
        theta = numpyro.deterministic("theta", Q @ z)

        sigma_beta = numpyro.sample("sigma_beta", dist.LogNormal(prior_log_s, 0.5))
        sigma_omega = numpyro.sample("sigma_omega", dist.LogNormal(prior_log_s, 0.5))

        for stat in stats:
            idx = jnp.array([model_index[m] for m in stat.models])
            mu_e = numpyro.sample(f"mu__{stat.name}", dist.Normal(float(np.mean(stat.y_hat)), 1.0))
            beta_tilde = numpyro.sample(f"beta_tilde__{stat.name}", dist.HalfNormal(1.0))
            omega_tilde = numpyro.sample(f"omega_tilde__{stat.name}", dist.HalfNormal(1.0))
            beta_e = numpyro.deterministic(f"beta__{stat.name}", sigma_beta * beta_tilde)
            omega_e = numpyro.deterministic(f"omega__{stat.name}", sigma_omega * omega_tilde)

            theta_e = theta[idx]
            mean_e = mu_e - beta_e * theta_e
            cov_e = omega_e**2 * jnp.eye(len(stat.models)) + jnp.array(stat.V)
            numpyro.sample(
                f"y__{stat.name}",
                dist.MultivariateNormal(mean_e, covariance_matrix=cov_e),
                obs=jnp.array(stat.y_hat),
            )

    return model
