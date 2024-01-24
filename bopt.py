import typer
from functools import partial

from typing import (
    Callable,
)

import jax.numpy as jnp
import jax.random as jr

from scipy.optimize import minimize

import matplotlib.pyplot as plt

from gp import GP
from kernels import *
from acquisition import *

# TODO : add functionality for adjusting the explore-exploit trade-off parameter
# TODO : replace all scipy methods with their jax counterparts
# TODO : clean up code + add comments + refine structure


class BayesianOptimization:
    def __init__(
        self,
        surrogate: Callable,
        acquisition: Callable,
        domain: tuple,
        dim: int,
        key: jr.PRNGKey = jr.PRNGKey(0),
        n_restarts: int = 25,
        X_init=None,
    ):
        self.surrogate = surrogate
        self.acquisition = acquisition
        self.domain = domain
        self.dim = dim
        self.key = key
        self.n_restarts = n_restarts
        self.X_init = X_init

    def propose(self):
        def objective(X_query):
            X_query = X_query.reshape(
                1, -1
            )  # surrogate require dim (N X D) where N = 1 here
            return -self.acquisition(X_query, self.surrogate)

        self.key, *_ = jr.split(self.key)  # rotate keys
        width = self.domain[1] - self.domain[0]
        bounds = jnp.array([self.domain[0], self.domain[1]]).T
        min_val = 1
        min_x = None
        # if surrogate model has not seen any data, propose the first observation uniformly
        if not hasattr(self.surrogate, "X_train"):
            if self.X_init == None:
                return (
                    jr.uniform(self.key, shape=(1, self.dim)) * width + self.domain[0]
                )
            else:
                return self.X_init
        else:
            # uniformly choose starting positions for optimization of the acquisition function
            # within domain; step 1 for global optimization
            positions = (
                jr.uniform(self.key, shape=(self.n_restarts, self.dim)) * width
                + self.domain[0]
            )
            # for every starting position, perform constrained optimization
            for x0 in positions:
                res = minimize(objective, x0=x0, bounds=bounds, method="L-BFGS-B")
                if res.fun < min_val:
                    min_val = res.fun
                    min_x = res.x
            return min_x.reshape(1, -1)

    def update(self, X, y, optimize: bool = False):
        # update surrogate model with newly observed data
        self.surrogate = self.surrogate.update(X, y)
        if optimize:
            # update kernel hyperparams
            self.surrogate = self.surrogate.optimize()

    def plot(self):
        self.surrogate.plot()

    def get_best(self):
        if not hasattr(self.surrogate, "y_train"):
            raise ValueError("Surrogate model has not be fitted.")
        ind = jnp.argmax(self.surrogate.y_train)
        return self.surrogate.X_train[ind]


def main(
    seed: int = 0,
    dim: int = 1,
    n_iter: int = 5,
    noise: float = 0.2,
    l: float = 1.0,
    sigma: float = 1.0,
    optimize: bool = False,
    plot: bool = False,
):
    if dim == 1:
        # true function
        true_f = lambda x: (-jnp.sin(3 * x) - x**2 + 0.7 * x).flatten()
        # mean function
        mean_f = lambda x: jnp.zeros_like(x).flatten()

        # domain
        domain = (jnp.array([-1.0]), jnp.array([2.0]))

        # init kernel
        kernel = SquaredExponential(
            theta=jnp.array([l, sigma]), bounds=jnp.array([[0.001, 100], [0.001, 100]])
        )

        key = jr.PRNGKey(seed)

        # init GP
        gp = GP(mean_f, kernel, noise, key, domain, true_f)

        # set up BayesOpt model
        model = BayesianOptimization(gp, expected_improvement, domain, dim=1)

        for i in range(n_iter):
            # ask model for point in X space
            X_next = model.propose()
            # evaluate point in X space
            y_next = true_f(X_next) + jr.normal(model.key) * noise
            # update model
            model.update(X_next, y_next, (i > 2) and optimize)

            if plot:
                model.plot()

        if plot:
            plt.show()
        print(model.get_best())
    else:
        true_f = lambda x: jnp.sum(x).flatten()
        mean_f = lambda x: jnp.zeros(x.shape[0])
        # domain
        domain = (jnp.array([-1.0] * dim), jnp.array([2.0] * dim))

        # init kernel
        kernel = SquaredExponential(theta=jnp.array([l, sigma]))

        # init GP
        gp = GP(mean_f, kernel, noise, seed, domain, None)

        # set up BayesOpt model
        model = BayesianOptimization(gp, expected_improvement, domain, dim=dim)

        for i in range(n_iter):
            # ask model for point in X space
            X_next = model.propose()
            # evaluate point in X space
            y_next = true_f(X_next) + jr.normal(model.key) * noise
            # update model
            model.update(X_next, y_next, (i > 2) and optimize)
        print(model.get_best())


if __name__ == "__main__":
    typer.run(main)
