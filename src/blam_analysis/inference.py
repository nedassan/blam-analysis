import concurrent.futures
import logging as log
import multiprocessing
import os

import numpy as np

from blam_analysis.model import build_model

log.basicConfig(level=log.INFO, format="%(asctime)s %(levelname)s %(message)s")


def resolve_backend(backend: str) -> str:
    if backend != "auto":
        if backend not in ("cpu", "gpu"):
            raise ValueError(f"backend must be 'auto', 'cpu', or 'gpu', got {backend!r}")
        return backend
    import jax

    return "gpu" if jax.default_backend() == "gpu" else "cpu"


def run_mcmc(model, seed: int, num_warmup: int, num_samples: int, num_chains: int, chain_method: str = "parallel"):
    import jax.random as random
    from numpyro.infer import MCMC, NUTS

    kernel = NUTS(model)
    mcmc = MCMC(
        kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains, chain_method=chain_method
    )
    mcmc.run(random.PRNGKey(seed))
    mcmc.print_summary(exclude_deterministic=False)
    flat_samples = mcmc.get_samples()
    grouped_samples = mcmc.get_samples(group_by_chain=True) if num_chains > 1 else None
    return flat_samples, grouped_samples


def run_single_chain(model, seed: int, num_warmup: int, num_samples: int) -> dict:
    import jax.random as random
    from numpyro.infer import MCMC, NUTS

    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=1, progress_bar=False)
    mcmc.run(random.PRNGKey(seed))
    return {k: np.asarray(v) for k, v in mcmc.get_samples().items()}


def combine_chains(chain_samples: list) -> tuple:
    flat_samples = {k: np.concatenate([c[k] for c in chain_samples], axis=0) for k in chain_samples[0]}
    grouped_samples = {k: np.stack([c[k] for c in chain_samples], axis=0) for k in chain_samples[0]}
    return flat_samples, grouped_samples


def _cpu_worker(global_models, stats, prior_log_s, num_warmup, num_samples, seed) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")

    model = build_model(global_models, stats, prior_log_s)
    return run_single_chain(model, seed, num_warmup, num_samples)


def _fit_cpu_multiprocess(
    global_models: list, stats: list, prior_log_s: float,
    num_warmup: int, num_samples: int, num_chains: int, seed: int, n_workers: int = None,
) -> tuple:
    n_workers = n_workers or min(num_chains, os.cpu_count() or 1)
    log.info(f"Fitting {num_chains} chain(s) across {n_workers} process(es) (CPU backend)")
    ctx = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
        futures = [
            pool.submit(_cpu_worker, global_models, stats, prior_log_s, num_warmup, num_samples, seed + i)
            for i in range(num_chains)
        ]
        chain_samples = [f.result() for f in futures]
    return combine_chains(chain_samples)


def fit_posterior(
    global_models: list, stats: list, prior_log_s: float,
    num_warmup: int = 1000, num_samples: int = 2000, num_chains: int = 4,
    seed: int = 0, backend: str = "auto", n_workers: int = None,
) -> tuple:
    backend = resolve_backend(backend)
    if backend == "gpu":
        log.info(f"Fitting {num_chains} chain(s) vectorized on GPU (one process)")
        model = build_model(global_models, stats, prior_log_s)
        return run_mcmc(model, seed, num_warmup, num_samples, num_chains, chain_method="vectorized")
    return _fit_cpu_multiprocess(global_models, stats, prior_log_s, num_warmup, num_samples, num_chains, seed, n_workers)
