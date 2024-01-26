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
