import jax.numpy as jnp
from jax.scipy.stats import norm

# TODO : clean up code + add comments + refine structure


def expected_improvement(X_query, surrogate, xi: float = 2.0):
    """
    Description:
        Computes the expected improvement (ei) at points X_query
    given samples (X, y) using a GP surrogate model (gp).

    Inputs:
        X_query: Locations to query (M x D).
        surrogate: surrogate regressor already fitted to samples (X, y).
        xi: Exploitation-exploration trade-off parameter.

    Returns:
        Expected improvement at points X_query.
    """
    if not hasattr(surrogate, "X_train"):
        raise ValueError("Surrogate Model has not been fitted to any data.")
    mu, _, sigma = surrogate.predict(X_query)
    mu_sample, *_ = surrogate.predict(surrogate.X_train)

    mu_sample_opt = jnp.max(mu_sample)

    imp = mu - mu_sample_opt - xi
    Z = imp / sigma
    ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei = ei.at[sigma == 0.0].set(0.0)

    return ei
