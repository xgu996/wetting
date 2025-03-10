# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 20:30:53 2023

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

pde = DropletLattice(domain=domain, theta = 70)

space = LagrangeFiniteElementSpace(pde.mesh, p=1)
p = space.interpolation_points()
solver = ChisdAllenCahn(pde)

gdof = space.number_of_global_dofs()
u = space.function()

data = loadmat("./solution_data/0001_DDDD.mat")
u[:] = data["u"].flatten()

mass = solver.b @ u

print(mass)

x = p[..., 0]
y = p[..., 1]
threshold = (x > 1.2)
u[threshold] = -1

one = np.ones(gdof, dtype =np.float64)
u -= (solver.b @ u - mass)/(solver.b @ one)

H = solver.hess_fenergy(u)

eval, evec = solver.calculate_eigs(H, n=20)
print(eval[0:3])

plt.tricontourf(p[..., 0], p[..., 1], u.tolist(), cmap="jet")