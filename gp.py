import typer

import numpy as np

import jax.numpy as jnp
import jax.random as jr
from jax.numpy.linalg import cholesky
from jax.scipy.linalg import cho_solve, solve_triangular

from scipy.optimize import minimize

from functools import partial

from typing import (
    Callable,
)

import matplotlib.pyplot as plt

from kernels import *

# TODO : clean up code + add comments + refine structure


class GP:
    def __init__(
        self,
        mean: Callable,
        kernel: object,
        noise: float = 0.0,  # noise in training
        key: jr.PRNGKey = jr.PRNGKey(0),  # random seed
        domain: tuple = (0, 1),  # X-range to plot
        f: Callable = None,  # true function
        n_test: int = 100,  # test set
        n_sample: int = 10,  # training set
    ):
        self.mean = mean
        self.kernel = kernel
        self.sigma = noise**2.0
        self.sigma_bounds = (0.0, 1e4)
        self.key = key
        self.domain = domain
        self.f = f
        self.n_test = n_test
        self.n_sample = n_sample

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        K = self.kernel.forward(X, X)
        i = jnp.diag_indices_from(K)
        K = K.at[i].set(K[i] + self.sigma)
        self.L = cholesky(K)
        if jnp.isnan(self.L).any() or jnp.isinf(self.L).any():
            raise ValueError("Non-PSD Cholesky. Try increasing sigma.")
        self.alpha = cho_solve((self.L, True), y - self.mean(X))
        return self

    def update(self, X, y):
        def set_L(n, k, lam1, lam2):
            L = jnp.zeros((n + k, n + k))
            L = L.at[:n, :n].set(self.L)
            L = L.at[n:, :n].set(lam1)
            L = L.at[n:, n:].set(lam2)
            return L

        if not hasattr(self, "X_train"):  # unfitted, nothing to update
            return self.fit(X, y)
        n, k = self.L.shape[0], X.shape[0]
        X1 = self.kernel.forward(X, self.X_train)
        X2 = self.kernel.forward(X, X)
        i = jnp.diag_indices_from(X2)
        X2 = X2.at[i].set(X2[i] + self.sigma)
        lam1 = solve_triangular(self.L, X1.T, lower=True, check_finite=False).T
        lam2 = cholesky(X2 - lam1 @ lam1.T)
        self.L = set_L(n, k, lam1, lam2)
        self.X_train = jnp.concatenate((self.X_train, X), axis=0)
        self.y_train = jnp.concatenate((self.y_train, y), axis=0)
        self.alpha = cho_solve((self.L, True), self.y_train - self.mean(self.X_train))
        return self

    def predict(self, X):
        if not hasattr(self, "X_train"):  # unfitted, predict based on GP prior
            f = self.mean(X)
            cov = self.kernel.forward(X, X)
            std = jnp.sqrt(jnp.diag(cov))
            return f, cov, std
        K = self.kernel.forward(X, self.X_train)
        f = self.mean(X) + K @ self.alpha
        v = solve_triangular(self.L, K.T, lower=True, check_finite=False)
        cov = self.kernel.forward(X, X) - v.T @ v
        std = jnp.sqrt(jnp.diag(cov))
        return f, cov, std

    def marginal_log_likelihood(self, theta=None):
        # TODO : check + supplement einsum computations
        n = self.X_train.shape[0]
        if theta is None:
            return (
                -0.5 * (self.y_train - self.mean(self.X_train)).T @ self.alpha
                - jnp.log(jnp.diag(self.L)).sum()
                - 0.5 * n * jnp.log(2.0 * jnp.pi)
            )
        else:
            kernel = self.kernel.copy(theta)
            # TODO: finish. add gradients for the noise. update the kernel with theta, etc.
            K = kernel.forward(self.X_train, self.X_train)
            # K_grad = kernel.backward(self.X_train, self.X_train)  # (N x N x D)
            i = jnp.diag_indices_from(K)
            K = K.at[i].set(K[i] + self.sigma)
            L = cholesky(K)
            if jnp.isnan(L).any() or jnp.isinf(L).any():
                raise ValueError("Non-PSD Cholesky. Try increasing sigma.")
            alpha = cho_solve((L, True), self.y_train - self.mean(self.X_train))
            log_likelihood = (
                -0.5 * (self.y_train - self.mean(self.X_train)).T @ alpha
                - jnp.log(jnp.diag(L)).sum()
                - 0.5 * n * jnp.log(2.0 * jnp.pi)
            )
            # K_inv = cho_solve((L, True), jnp.eye(n))
            # log_likelihood_grad = 0.5 * jnp.diagonal(
            #     (alpha @ alpha.T - K_inv) @ K_grad
            # ).sum(axis=-1)
            return log_likelihood
            # return log_likelihood, log_likelihood_grad

    def optimize(self):
        """Derivative-free optimization of marginal log likelihood via Nelder-Mead."""

        def obj_fun(theta):
            ll = self.marginal_log_likelihood(theta)
            return -ll

        n_restarts = 10
        best_res = jnp.inf
        best_theta = self.kernel.theta
        for _ in range(n_restarts):
            # run minimizer
            x0 = np.random.uniform(
                low=self.kernel.bounds[:, 0], high=self.kernel.bounds[:, 1]
            )
            bounds = self.kernel.bounds
            opt_res = minimize(
                obj_fun, x0, method="Nelder-Mead", bounds=bounds, tol=1e-6
            )
            if opt_res.fun < best_res:
                best_res = opt_res.fun
                best_theta = opt_res.x
        self.kernel.theta = best_theta
        # save pre-computed cholesky and alpha
        K = self.kernel.forward(self.X_train, self.X_train)
        i = jnp.diag_indices_from(K)
        K = K.at[i].set(K[i] + self.sigma)
        self.L = cholesky(K)
        if jnp.isnan(self.L).any() or jnp.isinf(self.L).any():
            raise ValueError("Non-PSD Cholesky. Try increasing sigma.")
        self.alpha = cho_solve((self.L, True), self.y_train - self.mean(self.X_train))
        return self

    def plot(self, plot_samples: bool = False, title: str = None):
        DISTS = ["Prior", "Posterior"]
        if plot_samples:
            fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(6, 6))
        else:
            fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(6, 3))
        X_test = jnp.linspace(self.domain[0], self.domain[1], self.n_test).reshape(
            -1, 1
        )
        mean, cov, std = self.predict(X_test)
        # Plot the distribution of the function (mean, covariance)
        if self.f is not None:
            ax1.plot(X_test, self.f(X_test), "b--", label="$sin(x)$")
        ax1.plot(X_test, mean, "r-", lw=2, label="$\mu_{2|1}$")
        ax1.fill_between(
            X_test.flatten(),
            mean - 2 * std,
            mean + 2 * std,
            color="red",
            alpha=0.15,
            label="$2 \sigma_{2|1}$",
        )
        if hasattr(self, "X_train"):
            ax1.plot(
                self.X_train, self.y_train, "ko", linewidth=2, label="$(x_1, y_1)$"
            )
        ax1.set_xlabel("$x$", fontsize=13)
        ax1.set_ylabel("$y$", fontsize=13)
        ax1.set_title(
            f'Model fit ({title if title is not None else (DISTS[1] if hasattr(self, "X_train") else DISTS[0])})'
        )
        # TODO : add in a smarter way of controlling the y-axis range
        ax1.axis([self.domain[0], self.domain[1], -120, 0])
        ax1.legend()
        if plot_samples:
            f_sample = jr.multivariate_normal(
                self.key, mean, cov, method="svd", shape=(self.n_sample,)
            )
            # Plot some samples from this function
            ax2.plot(X_test, f_sample.T, "-")
            ax2.set_xlabel("$x$", fontsize=13)
            ax2.set_ylabel("$y$", fontsize=13)
            ax2.set_title(f"{self.n_sample} different function realizations")
            ax1.axis([self.domain[0], self.domain[1], -3, 3])
            ax2.set_xlim([self.domain[0], self.domain[1]])
        plt.tight_layout()
        return fig


def main(
    seed: int = 0,
    n_train: int = 10,
    n_test: int = 75,
    n_sample: int = 10,
    noise: float = 0.2,
    sequential: bool = False,
    plot_posterior: bool = False,
    plot_prior: bool = False,
    optimize: bool = False,
    verbose: bool = False,
):
    # true function
    true_f = lambda x: (jnp.sin(x)).flatten()
    # mean function
    mean_f = lambda X: jnp.zeros_like(X).flatten()
    # domain
    domain = (-6, 6)
    # keys
    key = jr.PRNGKey(seed)
    key1, key2 = jr.split(key, 2)

    # Sample observations (X1, y1) on the function
    X_train = (domain[1] - domain[0]) * jr.uniform(key1, shape=(n_train, 1)) + domain[0]
    y_train = true_f(X_train) + jr.normal(key2, shape=(n_train,)) * noise

    # init kernel
    kernel = SquaredExponential(
        theta=jnp.array([1.0, 1.0]), bounds=jnp.array([[0.001, 100], [0.001, 100]])
    )

    # init GP
    gp = GP(mean_f, kernel, noise, key, domain, true_f, n_test, n_sample)

    if plot_prior:
        gp.plot(plot_samples=True)

    # train
    if sequential:
        for i, (X, y) in enumerate(zip(X_train, y_train)):
            X = jnp.expand_dims(X, axis=1)
            y = y.reshape(
                1,
            )
            # sequentially add data points + re-fit
            gp.update(X, y)

            if verbose:
                print(
                    f"Marginal Log-Likelihood after datum {i+1} = {gp.marginal_log_likelihood():.3f}"
                )
            if optimize:
                if verbose:
                    print(f"Optimizing kernel parameters....")
                gp = gp.optimize()
            if plot_posterior:
                gp.plot(
                    plot_samples=True,
                    title=f'Posterior {"w/ OPT" if optimize else "w/o OPT"}; step {i+1}',
                )

    else:
        gp = gp.fit(X_train, y_train)

        if optimize:
            if verbose:
                print(f"Optimizing kernel parameters....")
            if plot_posterior:
                gp.plot(plot_samples=True, title="Posterior No-Opt")
            gp = gp.optimize()

    if verbose:
        print(f"Final Marginal Log-Likelihood = {gp.marginal_log_likelihood():.3f}")

    if plot_posterior:
        gp.plot(plot_samples=True)

    if plot_posterior or plot_prior:
        plt.show()


if __name__ == "__main__":
    typer.run(main)
