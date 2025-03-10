# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 09:19:08 2023

@author: GDX666
"""

import sys

sys.path.append("..")

from fealpy.functionspace import LagrangeFiniteElementSpace
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat, savemat

from chisd_allencahn import ChisdAllenCahn
from mesh_generator import DomainParallelogramBottom
from model_data import DropletLattice


box = (0, 1.6, 0, 0.7)
bottom_gap = 0.08
shape_parameter = (0.06, bottom_gap * 0.8, 40)
domain = DomainParallelogramBottom(
    box=box, shape_parameter=shape_parameter, bottom_gap=bottom_gap
)

pde = DropletLattice(theta=102, mesh_file="./mesh_data/test1.vol")

space = LagrangeFiniteElementSpace(pde.mesh, p=1)
p = space.interpolation_points()
solver = ChisdAllenCahn(pde)

gdof = space.number_of_global_dofs()
u = space.function()

data = loadmat("./solution_data/001_DDU.mat")
u[:] = data["u"].flatten()

mass = solver.b @ u

print(mass)

x = p[..., 0]
y = p[..., 1]


# u[(x >= 0.7) & (x <= 0.92) & (y <= 0.2) & (y >= 0.1)] = 1
# u[(x >= 0.78) & (x <= 0.9) & (y <= 0.25) & (y >= 0)] = 1
# u[(x >= 0.7) & (x <= 0.9) & (y <= 0.2) & (y >= 0.1)] = 1
# u[(x >= 0.75) & (x <= 0.873) & (y <= 0.02) & (y >= 0)] = -1

# u[(x >= 0.291) & (x <= 0.355) & (y >= 0)] = -1
# u[(x >= 0.361) & (x <= 0.393) & (y >= 0) & (y<=0.1)] = -1

# u[x >= 0.75] = -1
# u[(y >= 0) & (y <= 0.1) & (x >= 0.64)] = -1

u[(x >= 0.6) & (x<=0.74) & (y>=0.02) & (y<0.08)] = 1

one = np.ones(gdof, dtype=np.float64)
u -= (solver.b @ u - mass) / (solver.b @ one)

plt.figure()
plt.tricontourf(
    p[..., 0], p[..., 1], u.tolist(), triangles=pde.mesh.entity("cell"), cmap="jet"
)

# -------------------------
H = solver.hess_fenergy(u)
start = 40
end = 40
eval, evec = solver.calculate_eigs(H, n=end)
print(eval[0:3])
perturb_direction = evec[:, start:end]
# perturb_direction = np.hstack(
#     (evec[:, 0:1].reshape(gdof, 1), evec[:, start:end])
# )
up = solver.calculate_perturb(uh=u, a=0, direction=perturb_direction)
plt.figure()
plt.tricontourf(
    p[..., 0], p[..., 1], up.tolist(), triangles=pde.mesh.entity("cell"), cmap="jet"
)
print(solver.b @ u)
print(solver.b @ up)
print(solver.b @ u)

# savemat('./solution_data/0001_UUUU_R.mat', {"u":up.tolist()})
# savemat('./solution_data/0000_UUUU_R.mat', {"u":up.tolist()})
# savemat("./solution_data/100_UUU.mat", {"u": up.tolist()})
savemat("./solution_data/sd2_w.mat", {"u": up.tolist()})

plt.show()
