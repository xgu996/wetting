import sys

import numpy as np
from fealpy.decorator import cartesian

from src.mesh_generator import (DomainParallelogramBottom,
                                DomainRectangularBottom, generate_mesh)


class DropletLattice:
    def __init__(
        self,
        epsilon=1e-2,
        theta=102,
        domain=DomainRectangularBottom(),
        mesh_file=None,
        # domain=DomainParallelogramBottom(),
    ):
        self.epsilon = epsilon
        self.domain = domain
        geo_info = self.domain.geometry_info()
        self.theta = theta * np.pi / 180
        self.mesh = generate_mesh(geo_info, maxh=0.01, mesh_file=mesh_file)

    @cartesian
    def initial_value(self, p, center=0):
        u0 = -np.ones(p.shape[0], dtype=np.float64)
        x = p[..., 0]
        y = p[..., 1]
        arc = 120 * np.pi / 180
        # arc = 141 * np.pi / 180

        # center = (self.domain.box[0] + self.domain.box[1]) / 2
        xl = center - 3 * 0.08 / 2 - 3 * 0.06 / 2
        xr = center + 3 * 0.08 / 2 + 3 * 0.06 / 2
        h = self.domain.shape_parameter[1]
        xc = (xl + xr) / 2
        yc = h - (xr - xl) / np.tan(arc) / 2
        r = (xr - xl) / np.sin(arc) / 2
        u0[(y >= h) & ((x - xc) ** 2 + (y - yc) ** 2 <= r**2)] = 1
        return u0

    @cartesian
    def is_robin_boundary(self, p):
        bottom_height = self.domain.shape_parameter[1]
        x = p[..., 0]
        y = p[..., 1]
        flag = (x > 0.0) & (x < self.domain.box[1]) & (y <= bottom_height)
        return flag

    def f(self, u):
        return (u**2 - 1) ** 2 / 4

    def diff_f(self, u, pow=1):
        if pow == 1:
            return u**3 - u
        elif pow == 2:
            return 3 * u**2 - 1

    def g(self, u):
        c = np.cos(self.theta)
        return -c * (3 * u - u**3) * np.sqrt(2) / 6

    def diff_g(self, u, pow):
        c = np.cos(self.theta)
        if pow == 1:
            return -c * (1 - u**2) * np.sqrt(2) / 2
        elif pow == 2:
            return np.sqrt(2) * c * u


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from fealpy.functionspace import LagrangeFiniteElementSpace

    pde = DropletLattice()
    pde.mesh.add_plot(plt)
    space = LagrangeFiniteElementSpace(pde.mesh, 1)
    p = space.interpolation_points()
    u = space.function()
    u[:] = pde.initial_value(p)
    u.add_plot(plt, cmap="jet")
    plt.show()


# class DropletSquareDomain:
#     def __init__(self, epsilon=1e-2, theta = 110):
#         self.epsilon = epsilon
#         mf = MeshFactory
#         self.mesh = mf.boxmesh2d([0, 1, 0, 1], nx=50, ny=50, meshtype="tri")
#         self.theta = theta * np.pi / 180

#     @cartesian
#     def initial_value(self, p):
#         u = -np.ones(p.shape[0], dtype=np.float64)
#         x = p[..., 0]
#         y = p[..., 1]
#         u[((x - 0.5) ** 2 + y**2 <= 0.2**2) & (y >= 0)] = 1
#         return u

#     @cartesian
#     def is_robin_boundary(self, p):
#         y = p[..., 1]
#         return y == 0.0

#     def f(self, u):
#         return (u**2 - 1) ** 2 / 4

#     def diff_f(self, u, pow=1):
#         if pow == 1:
#             return u**3 - u
#         elif pow == 2:
#             return 3 * u**2 - 1

#     def g(self, u):
#         c = np.cos(self.theta)
#         return -c * (3 * u - u**3) * np.sqrt(2) / 6

#     def diff_g(self, u, pow):
#         c = np.cos(self.theta)
#         if pow == 1:
#             return -c * (1 - u**2) * np.sqrt(2) / 2
#         elif pow == 2:
#             return np.sqrt(2) * c * u


# class BenchMarkSquareDomain:
#     def __init__(self, epsilon=1e-2):
#         self.epsilon = epsilon
#         mf = MeshFactory
#         self.mesh = mf.boxmesh2d([0, 1, 0, 1], nx=50, ny=50, meshtype="tri")

#     @cartesian
#     def dirichlet(self, p):
#         gD = np.zeros(p.shape[0])
#         return gD

#     @cartesian
#     def is_robin_boundary(self, p):

#         return np.zeros(p.shape[0], dtype = np.bool_)

#     @cartesian
#     def initial_value(self,  p):
#         u = np.zeros(p.shape[0], dtype=np.float64)
#         x = p[..., 0]
#         y = p[..., 1]
#         u[(x >= 0.35) & (x <= 0.65) & (y >= 0.35) & (y <= 0.65)] = 1
#         return u

#     def f(self, u):
#         return u**2 * (u - 1) ** 2 / 2

#     def diff_f(self, u, pow=1):
#         if pow == 1:
#             return u * (u - 1) * (2 * u - 1)
#         elif pow == 2:
#             return 6 * u * (u - 1) + 1

#     def g(self, u):
#         return np.zeros(u.shape)

#     def diff_g(self, u, pow):
#         return np.zeros(u.shape)
