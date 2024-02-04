
<div align="center">

# `jaxproxqp` - ProxQP in Jax

[Installation](#installation) •
[Summary](#summary) •
[Quickstart](#quickstart)

</div>

## Installation
Install the latest development version from GitHub:
```bash
pip install git+https://github.com/oswinso/jaxproxqp.git
```

## Summary
We aim to solve Quadratic Programs (QP) of the form

$$
\begin{align}
\min_{x} &  ~\frac{1}{2}x^{T}Hx+g^{T}x \\
\text{s.t.} & ~A x = b \\
& ~l \leq C x \leq u \\
& ~l_{\mathrm{box}} \leq x \leq u_{\mathrm{box}}
\end{align}
$$

where $x \in \mathbb{R}^n$ is the optimization variable, and $H \in \mathcal{S}^n_+$ is a positive semidefinite matrix. We implement the [ProxQP](https://www.roboticsproceedings.org/rss18/p040.pdf) method in [JAX](https://github.com/google/jax). `jaxproxqp` is `vmap`-able and `jit`-able, allowing for _very fast_ solving of many QPs in parallel on GPU.

## Quickstart
Consider the following QP

$$
  \begin{array}{ll}
    \displaystyle \min_x & \frac{1}{2} x^T \begin{bmatrix}4 & 1\\\\ 1 & 2 \end{bmatrix} x + \begin{bmatrix}1 \\\\ 1\end{bmatrix}^T x \\
    \mathrm{s.t.} & \begin{bmatrix}1 \end{bmatrix} \leq \begin{bmatrix} 1 & 1\end{bmatrix} x \leq  \begin{bmatrix}2 \end{bmatrix} \\
    & \begin{bmatrix} 0 \\\\ 0\end{bmatrix} \leq x \leq \begin{bmatrix}0.7 \\\\ 0.7\end{bmatrix}
  \end{array}
$$

We can solve this QP in `jaxproxqp` as follows.
```python
import jax
import numpy as np

from jaxproxqp.jaxproxqp import JaxProxQP
from jaxproxqp.qp_problems import QPModel
from jaxproxqp.settings import Settings

# Enable 64 bit.
jax.config.update("jax_enable_x64", True)

H = np.array([[4, 1], [1, 2]])
g = np.array([1, 1])

C = np.array([[1, 1]])
l = np.array([1.0])
u = np.array([2.0])

l_box = np.array([0.0, 0.0])
u_box = np.array([0.7, 0.7])

qp = QPModel.create(H=H, g=g, C=C, u=u, l=l, u_box=u_box, l_box=l_box)

settings = Settings.default_float64()
solver = JaxProxQP(qp, settings)
sol = solver.solve()

print("Objective Value                : ", sol.obj_value)
print("Solution                       : ", sol.x)
print("Inequality Lagrange Multipliers: ", sol.z)
print()
print("Cx                             : {} <= {} <= {}".format(l, C @ sol.x, u))
print(" x                             : {} <= {} <= {}".format(l_box, sol.x, u_box))
```
This example is in [examples/solve_qp_example.py](examples/solve_qp_example.py)

## Citing `jaxproxqp`
If you use `jaxproxqp` in your work, consider citing this software repository:
```bibtex
@software{so2024jaxproxqp,
  title = {jaxproxqp},
  author = {So, Oswin},
  url = {http://github.com/oswinso/jaxproxqp},
  year = {2024},
}
```
Also consider citing the [ProxQP](https://www.roboticsproceedings.org/rss18/p040.pdf) paper:
```bibtex
@inproceedings{bambade2022prox,
  title={Prox-qp: Yet another quadratic programming solver for robotics and beyond},
  author={Bambade, Antoine and El-Kazdadi, Sarah and Taylor, Adrien and Carpentier, Justin},
  booktitle={RSS 2022-Robotics: Science and Systems},
  year={2022}
}
```

## Projects using `jaxproxqp`
This codebase has been used in the following projects:
- So et al., “How to train your neural control barrier function: Learning
safety filters for complex input-constrained systems,” [website](https://mit-realm.github.io/pncbf/) | [paper](https://arxiv.org/abs/2310.15478) 
- Zhang et al., "GCBF+: A Neural Graph Control Barrier Function Framework for Distributed Safe Multi-Agent Control," [website](https://mit-realm.github.io/gcbfplus-website) | [paper](https://arxiv.org/abs/2401.14554)
