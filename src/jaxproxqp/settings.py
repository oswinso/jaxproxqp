import jax.numpy as jnp
from attrs import define
from loguru import logger

from jaxproxqp.utils.jax_types import FloatScalar

_HAS_LOGGED = False


@define
class Settings:
    alpha_bcl: float = 0.1
    beta_bcl: float = 0.9

    mu_min_eq: float = 1e-9
    mu_min_in: float = 1e-8
    mu_max_eq_inv: float = 1e9
    mu_max_in_inv: float = 1e8

    mu_update_factor: float = 0.1
    mu_update_inv_factor: float = 10.0

    cold_reset_mu_eq: float = 1.0 / 1.1
    cold_reset_mu_in: float = 1.0 / 1.1
    cold_reset_mu_eq_inv: float = 1.1
    cold_reset_mu_in_inv: float = 1.1

    eps_abs: float = 1.0e-5
    # Smallest possible value for bcl_eta_in
    eps_in_min: float = 1e-9

    pri_res_thresh_abs: float = 1e-5
    dua_res_thresh_abs: float = 1e-5
    dua_gap_thresh_abs: float | None = 1e-5

    # Threshold for early exit from inner loop for a step size that is too small.
    step_size_thresh: float = 1e-8

    max_iter: int = 100
    max_iter_in: int = 15
    safe_guard: int = 100

    max_iterative_refine: int = 2

    preconditioner_max_iter: int = 10

    alpha_gpdal: FloatScalar = 0.95

    # Use f64 for computing the residuals for iterative refinement.
    use_f64_refine: bool = False

    # If true, print during execution.
    verbose: bool = False

    @staticmethod
    def default():
        float_type = jnp.ones(0).dtype
        global _HAS_LOGGED

        if float_type == jnp.float32:
            if not _HAS_LOGGED:
                logger.debug("Using default settings for float32")
                _HAS_LOGGED = True

            return Settings.default_float32()
        elif float_type == jnp.float64:
            if not _HAS_LOGGED:
                logger.debug("Using default settings for float64")
                _HAS_LOGGED = True

            return Settings.default_float64()
        else:
            raise ValueError(f"Unsupported float type: {float_type}")

    @staticmethod
    def default_float64() -> "Settings":
        return Settings(
            alpha_bcl=0.1,
            beta_bcl=0.9,
            mu_min_eq=1e-9,
            mu_min_in=1e-8,
            mu_max_eq_inv=1e9,
            mu_max_in_inv=1e8,
            #
            mu_update_factor=0.1,
            mu_update_inv_factor=10.0,
            #
            cold_reset_mu_eq=1.0 / 1.1,
            cold_reset_mu_in=1.0 / 1.1,
            cold_reset_mu_eq_inv=1.1,
            cold_reset_mu_in_inv=1.1,
            #
            eps_abs=1.0e-5,
            eps_in_min=1e-9,
            #
            pri_res_thresh_abs=1e-9,
            dua_res_thresh_abs=1e-9,
            # Threshold for early exit from inner loop for a step size that is too small.
            step_size_thresh=1e-8,
            #
            max_iter=100,
            max_iter_in=15,
            safe_guard=100,
            max_iterative_refine=2,
            #
            preconditioner_max_iter=10,
            #
            alpha_gpdal=0.95,
            use_f64_refine=False,
            verbose=False,
        )

    @staticmethod
    def default_float32() -> "Settings":
        return Settings(
            alpha_bcl=0.1,
            beta_bcl=0.9,
            mu_min_eq=1e-9,
            mu_min_in=1e-8,
            mu_max_eq_inv=1e9,
            mu_max_in_inv=1e8,
            #
            mu_update_factor=0.1,
            mu_update_inv_factor=10.0,
            #
            cold_reset_mu_eq=1.0 / 1.1,
            cold_reset_mu_in=1.0 / 1.1,
            cold_reset_mu_eq_inv=1.1,
            cold_reset_mu_in_inv=1.1,
            #
            eps_abs=5.0e-5,
            eps_in_min=5.0e-5,
            #
            # pri_res_thresh_abs=8e-5,
            # pri_res_thresh_abs=5e-4,
            pri_res_thresh_abs=8e-4,
            dua_res_thresh_abs=1e-3,
            # Threshold for early exit from inner loop for a step size that is too small.
            step_size_thresh=1e-8,
            #
            max_iter=25,
            max_iter_in=15,
            safe_guard=18,
            max_iterative_refine=10,
            #
            preconditioner_max_iter=10,
            #
            alpha_gpdal=0.95,
            use_f64_refine=False,
            verbose=False,
        )

    @property
    def bcl_eta_ext_init(self):
        return 0.1**self.alpha_bcl
