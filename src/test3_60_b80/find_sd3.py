# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 09:19:08 2023

@author: GDX666
"""

import sys

sys.path.append("..")

import matplotlib.pyplot as plt
import numpy as np
from chisd_allencahn import ChisdAllenCahn
from fealpy.functionspace import LagrangeFiniteElementSpace
from mesh_generator import DomainParallelogramBottom
from model_data import DropletLattice
from scipy.io import loadmat, savemat

box = (0, 1.6, 0, 0.7)
bottom_gap = 0.08
shape_parameter = (0.06, bottom_gap, 80)
domain = DomainParallelogramBottom(
    box=box, shape_parameter=shape_parameter, bottom_gap=bottom_gap
)

pde = DropletLattice(domain=domain, theta=60, mesh_file="./mesh_data/test3_80_b80.vol")

space = LagrangeFiniteElementSpace(pde.mesh, p=1)
p = space.interpolation_points()
solver = ChisdAllenCahn(pde)

gdof = space.number_of_global_dofs()
u = space.function()

data = loadmat("./solution_data/000_DDD.mat")
u[:] = data["u"].flatten()

mass = solver.b @ u

print(mass)

x = p[..., 0]
y = p[..., 1]

# 1111
# threshold = (x > 0.8) & (x < 1.2) & (y >= 0) & (y <= bottom_gap)
# u[threshold] = -1
# u[(x > 0.92) & (y >= bottom_gap) & (y <= 0.7)] = -1
# u[(x > 0.91) & (x < 0.935) & (y >= bottom_gap) & (y <= 0.12)] = 1

# 2222
u[(x > 0.39) & (x < 0.6) & (y >= bottom_gap) & (y <= 0.2)] = 1

one = np.ones(gdof, dtype=np.float64)
u -= (solver.b @ u - mass) / (solver.b @ one)

plt.figure()
plt.tricontourf(p[..., 0], p[..., 1], u.tolist(), cmap="jet")

# -------------------------
H = solver.hess_fenergy(u)
start = 10
end = 50
eval, evec = solver.calculate_eigs(H, n=end)
print(eval[0:3])
perturb_direction = evec[:, start:end]
# perturb_direction = np.hstack(
#     (evec[:, 0:1].reshape(gdof, 1), evec[:, start:end])
# )
up = solver.calculate_perturb(uh=u, a=0.01, direction=perturb_direction)
plt.figure()
plt.tricontourf(
    p[..., 0], p[..., 1], up.tolist(), triangles=pde.mesh.entity("cell"), cmap="jet"
)
print(solver.b @ u)
print(solver.b @ up)

print(solver.b @ u)
