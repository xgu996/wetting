# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 20:09:25 2023

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
shape_parameter = (0.06, bottom_gap * 0.8, 40)
domain = DomainParallelogramBottom(
    box=box, shape_parameter=shape_parameter, bottom_gap=bottom_gap
)

pde = DropletLattice(domain=domain, theta=97)

space = LagrangeFiniteElementSpace(pde.mesh, p=1)
p = space.interpolation_points()
solver = ChisdAllenCahn(pde)

gdof = space.number_of_global_dofs()
u = space.function()

data = loadmat("./solution_data/000_UUU.mat")
# data = loadmat("./solution_data/00_UU_R.mat")
# data = loadmat("./solution_data/100_UUU_L.mat")
# data = loadmat("./solution_data/00_UU_L.mat")
u[:] = data["u"].flatten()

H = solver.hess_fenergy(u)

start = 20
end = 45

eval, evec = solver.calculate_eigs(H, n=end)

print(eval[0:3])
perturb_direction = evec[:, start:end]
# perturb_direction = np.hstack((evec[:, 0:1].reshape(gdof, 1), evec[:, start:end]))
up = solver.calculate_perturb(uh=u, a=0.02, direction=perturb_direction)
plt.tricontourf(p[..., 0], p[..., 1], up.tolist(), cmap="jet")
print(solver.b @ u)
print(solver.b @ up)
