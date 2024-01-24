import jax
import jax.numpy as jnp


def squared_exponential(xa, xb, l: float = 1.0, sigma: float = 1.0):
    """
    Squared Exponential Kernel.
    """
    # (L2 distance (Squared Euclidian))
    sq_norm = (
        -(1.0 / (2.0 * l**2.0))
        * jnp.linalg.norm(
            jnp.expand_dims(xa, axis=1) - jnp.expand_dims(xb, axis=0), axis=-1
        )
        ** 2.0
    )
    return (sigma**2.0) * jnp.exp(sq_norm)


class SquaredExponential:
    def __init__(
        self,
        theta: jax.Array = jnp.array([1.0, 1.0]),
        bounds: jax.Array = jnp.array([[1e-3, 1e3], [1e-3, 1e3]]),
    ):
        self.n_dim = 2  # number of non-fixed hyperparameters of this kernel
        if theta.shape[0] != self.n_dim:
            raise ValueError(
                "Theta shape does not match number of kernel hyperparameters."
            )
        if bounds.shape[0] != self.n_dim:
            raise ValueError("Theta bounds are inconsistent.")
        self.theta = theta
        self.bounds = bounds
        self.hyperparameters = ["l", "sigma"]

    def forward(self, xa, xb):
        # (L2 distance (Squared Euclidian))
        sq_norm = (
            jnp.linalg.norm(
                jnp.expand_dims(xa, axis=1) - jnp.expand_dims(xb, axis=0), axis=-1
            )
            ** 2.0
        )
        return (self.theta[1] ** 2.0) * jnp.exp(
            -(1.0 / (2.0 * self.theta[0] ** 2.0)) * sq_norm
        )

    def backward(self, xa, xb):
        # derivatives wrt the kernel parameters
        sq_norm = (
            jnp.linalg.norm(
                jnp.expand_dims(xa, axis=1) - jnp.expand_dims(xb, axis=0), axis=-1
            )
            ** 2.0
        )
        dKdl = (
            (self.theta[1] ** 2.0)
            * jnp.exp(-(1.0 / (2.0 * self.theta[0] ** 2.0)) * sq_norm)
            * sq_norm
            * (1.0 / self.theta[0] ** 3.0)  # * self.theta[0]
        )
        dKdsigma = (
            2.0
            * self.theta[1]
            * jnp.exp(
                -(1.0 / (2.0 * self.theta[0] ** 2.0)) * sq_norm
            )  # * self.theta[1]
        )
        return jnp.stack((dKdsigma, dKdl), axis=-1)

    def copy(self, theta):
        return SquaredExponential(theta)


class White:
    def __init__(
        self,
        theta: jax.Array = jnp.array([1.0]),
        bounds: jax.Array = jnp.array([[0.0, 1e5]]),
    ):
        self.n_dim = 1  # number of non-fixed hyperparameters of this kernel
        if theta.shape[0] != self.n_dim:
            raise ValueError(
                "Theta shape does not match number of kernel hyperparameters."
            )
        if bounds.shape[0] != self.n_dim:
            raise ValueError("Theta bounds are inconsistent.")
        self.theta = theta
        self.bounds = bounds
        self.hyperparameters = ["sigma"]

    def forward(self, xa, xb):
        if xa == xb:
            return self.theta[0] * jnp.eye(xa.shape[0])
        else:
            return jnp.zeros((xa.shape[0], xb.shape[0]))

    def backward(self, xa, xb):
        raise NotImplementedError

    def copy(self, theta):
        return White(theta)
