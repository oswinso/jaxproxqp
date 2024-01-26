from typing import NamedTuple

import jax.numpy as jnp

from jaxproxqp.qp_problems import QPBox
from jaxproxqp.qp_types import EMat, EVec, IMat, IVec, IXBool, IXVec, WVec, XMat, XVec
from jaxproxqp.utils.jax_types import FloatScalar


class Workspace(NamedTuple):
    # QP Storage.
    H_scaled: XMat
    g_scaled: XVec

    A_scaled: EMat
    C_scaled: IMat

    b_scaled: EVec
    u_scaled: IVec
    l_scaled: IVec

    u_box_scaled: XVec
    l_box_scaled: XVec

    i_scaled: IVec

    # Initial variable.
    x_prev: XVec
    y_prev: EVec
    z_prev: IXVec

    # Active set.
    active_set_up: IXBool
    active_set_low: IXBool
    active_ineqs: IXBool

    # First order residuals for line search.
    Hdx: XVec
    Cdx: IXVec
    Adx: EVec

    # Newton variables
    dw_aug: WVec
    rhs: WVec
    err: WVec

    alpha: FloatScalar

    dual_residual_scaled: XVec
    # (n_in + dim, ). Contains unscaled [ Cx; x ] after global_primal_residual.
    primal_residual_in_scaled_up: IXVec

    pri_res_in_sc_up_plus_alphaCdx: IXVec
    pri_res_in_sc_low_plus_alphaCdx: IXVec
    CTz: XVec

    @property
    def pri_res_in_sc_up_head(self) -> IVec:
        return self.primal_residual_in_scaled_up[: self.n_in]

    @property
    def pri_res_in_sc_up_tail(self) -> IXVec:
        s = self.n_in
        return self.primal_residual_in_scaled_up[s : s + self.dim]

    @property
    def active_set_up_head(self) -> IVec:
        return self.active_set_up[: self.n_in]

    @property
    def active_set_up_tail(self) -> IVec:
        s = self.n_in
        return self.active_set_up[s : s + self.dim]

    @property
    def active_set_low_head(self) -> IVec:
        return self.active_set_low[: self.n_in]

    @property
    def active_set_low_tail(self) -> IVec:
        s = self.n_in
        return self.active_set_low[s : s + self.dim]

    @property
    def dz(self) -> IXVec:
        return self.dw_aug[-(self.n_in + self.dim) :]

    @property
    def dw_aug_parts(self) -> tuple[XVec, EVec, IVec, XVec]:
        dx = self.dw_aug[: self.dim]
        s = self.dim
        dy = self.dw_aug[s : s + self.n_eq]
        s += self.n_eq
        dz_head = self.dw_aug[s : s + self.n_in]
        s += self.n_in
        dz_tail = self.dw_aug[s : s + self.dim]
        return dx, dy, dz_head, dz_tail

    @property
    def dim(self):
        return self.H_scaled.shape[0]

    @property
    def n_eq(self):
        return self.A_scaled.shape[0]

    @property
    def n_in(self):
        return self.C_scaled.shape[0]

    @staticmethod
    def create(qp: QPBox):
        x_prev = jnp.zeros(qp.dim)
        y_prev = jnp.zeros(qp.n_eq)
        z_prev = jnp.zeros(qp.n_in)

        active_set_up = jnp.zeros(qp.n_in + qp.dim, dtype=bool)
        active_set_low = jnp.zeros(qp.n_in + qp.dim, dtype=bool)
        active_ineqs = jnp.zeros(qp.n_in + qp.dim, dtype=bool)

        Hdx = jnp.zeros(qp.dim)
        Cdx = jnp.zeros(qp.n_in + qp.dim)
        Adx = jnp.zeros(qp.n_eq)
        dual_resid_scaled = jnp.zeros(qp.dim)
        primal_residual_in_scaled_up = jnp.zeros(qp.n_in + qp.dim)
        pri_res_in_sc_up_plus_alphaCdx = jnp.zeros(qp.n_in + qp.dim)
        pri_res_in_sc_low_plus_alphaCdx = jnp.zeros(qp.n_in + qp.dim)

        n_w = qp.dim + qp.n_eq + qp.n_in + qp.dim
        dw_aug = jnp.zeros(n_w)
        rhs = jnp.zeros(n_w)
        err = jnp.zeros(n_w)

        alpha = 0.0

        CTz = jnp.zeros(qp.dim)

        return Workspace(
            qp.H,
            qp.g,
            qp.A,
            qp.C,
            qp.b,
            qp.u,
            qp.l,
            qp.u_box,
            qp.l_box,
            qp.I,
            x_prev,
            y_prev,
            z_prev,
            active_set_up,
            active_set_low,
            active_ineqs,
            Hdx,
            Cdx,
            Adx,
            dw_aug,
            rhs,
            err,
            alpha,
            dual_resid_scaled,
            primal_residual_in_scaled_up,
            pri_res_in_sc_up_plus_alphaCdx,
            pri_res_in_sc_low_plus_alphaCdx,
            CTz,
        )
