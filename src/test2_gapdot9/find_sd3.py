# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 09:19:08 2023

@author: GDX666
"""

import sys
sys.path.append("..")
sys.path.append("../src")

from fealpy.functionspace import LagrangeFiniteElementSpace
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat, savemat

from chisd_allencahn import ChisdAllenCahn
from mesh_generator import DomainRectangularBottom
from model_data import DropletLattice


box = (0, 1.6, 0, 0.7)
bottom_gap = 0.09
shape_parameter = (0.045, 0.1)
domain = DomainRectangularBottom(
    box=box, shape_parameter=shape_parameter, bottom_gap=bottom_gap
)


pde = DropletLattice(domain=domain)

space = LagrangeFiniteElementSpace(pde.mesh, p=1)
p = space.interpolation_points()
solver = ChisdAllenCahn(pde)

gdof = space.number_of_global_dofs()
u = space.function()

data = loadmat("./solution_data/000_UDD.mat")
u[:] = data["u"].flatten()

mass = solver.b @ u

print(mass)

x = p[..., 0]
y = p[..., 1]
threshold = (x > 0.6) & (x <0.72) & (y > 0.02) & (y <= 0.1)
u[threshold] = 1

one = np.ones(gdof, dtype =np.float64)
u -= (solver.b @ u - mass)/(solver.b @ one)

plt.figure()
plt.tricontourf(p[..., 0], p[..., 1], u.tolist(), cmap = 'jet')

#-------------------------
H = solver.hess_fenergy(u)
start = 20
end = 45
eval, evec = solver.calculate_eigs(H, n=end)
print(eval[0:3])
perturb_direction = evec[:, start:end]
# perturb_direction = np.hstack(
#     (evec[:, 0:1].reshape(gdof, 1), evec[:, start:end])
# )
up = solver.calculate_perturb(uh=u, a=0.005, direction=perturb_direction)
plt.figure()
plt.tricontourf(p[..., 0], p[..., 1], up.tolist(), cmap="jet")
print(solver.b @ u)
print(solver.b @ up)

print(solver.b @ u)