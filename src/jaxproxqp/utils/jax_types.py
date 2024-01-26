from typing import Union

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int

Arr = Union[np.ndarray, Array]

AnyFloat = Float[Arr, "*"]
Shape = tuple[int, ...]

FloatScalar = float | Float[Arr, ""]
IntScalar = int | Int[Arr, ""]
BoolScalar = bool | Bool[Arr, ""]


def get_default_float_dtype():
    return jnp.zeros(0).dtype


def float32_is_default() -> bool:
    return get_default_float_dtype() == jnp.float32


def assert_x64_enabled():
    if jnp.zeros(0, dtype=jnp.float64).dtype != jnp.float64:
        raise ValueError(
            "jax_enable_x64 must be set to True! Note that we can still keep the default dtype "
            "as float32 by using jax_default_dtype_bits=32."
        )
