import ipdb
import jax
import jax.random as jr
import numpy as np

from jaxproxqp.precond.ruiz import Ruiz
from jaxproxqp.qp_problems import QPBox, QPModel
from jaxproxqp.utils.random_qp import strongly_convex_qp


def main():
    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_enable_x64", True)
    np.set_printoptions(linewidth=300)

    key = jr.PRNGKey(12345)
    sparsity_factor = 0.15
    strong_convexity_factor = 0.01

    dim, n_eq, n_in = 5, 5, 3

    qp: QPModel = strongly_convex_qp(key, dim, n_eq, n_in, sparsity_factor, strong_convexity_factor)

    qp_box = QPBox(qp.H, qp.g, qp.A, qp.C, qp.b, qp.u, qp.l, np.ones(1), qp.u_box, qp.l_box)
    ruiz = Ruiz.create(qp.dim, qp.n_eq, qp.n_in)
    ruiz_n, c, qp_n = ruiz.scale_qp(qp_box, max_iter=3)

    H, g, A, C, b, u, l, u_box, l_box = qp

    H_n = c * ruiz_n.delta_x[:, None] * H * ruiz_n.delta_x[None, :]
    g_n = c * ruiz_n.delta_x * g

    A_n = ruiz_n.delta_eq[:, None] * A * ruiz_n.delta_x[None, :]
    b_n = ruiz_n.delta_eq * b

    C_n = ruiz_n.delta_in[:, None] * C * ruiz_n.delta_x[None, :]
    u_n = ruiz_n.delta_in * u
    l_n = ruiz_n.delta_in * l

    u_box_n = ruiz_n.delta_box * u_box
    l_box_n = ruiz_n.delta_box * l_box

    np.testing.assert_allclose(H_n, qp_n.H)
    np.testing.assert_allclose(g_n, qp_n.g)

    np.testing.assert_allclose(A_n, qp_n.A)
    np.testing.assert_allclose(b_n, qp_n.b)

    np.testing.assert_allclose(C_n, qp_n.C)
    np.testing.assert_allclose(u_n, qp_n.u)
    np.testing.assert_allclose(l_n, qp_n.l)

    np.testing.assert_allclose(u_box_n, qp_n.u_box)
    np.testing.assert_allclose(l_box_n, qp_n.l_box)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
