import os
import sys

from fealpy.functionspace import LagrangeFiniteElementSpace
from numpy import load
from scipy.io import loadmat, savemat

sys.path.append("..")

from chisd_allencahn import ChisdAllenCahn
from mesh_generator import DomainRectangularBottom
from model_data import DropletLattice


class Test:
    def __init__(self, ipath, opath=None):
        self.ipath = ipath

        if opath is None:
            self.opath = self.ipath
        else:
            self.opath = opath

    def input_data(self):
        u0 = solver.space.function()
        idata = loadmat(self.ipath)
        u0[:] = idata["u"].flatten()
        return u0

    def export_data(self, uf, vf=None):
        if vf is None:
            savemat(self.opath, {"u": uf.tolist()})
        else:
            savemat(self.opath, {"u": uf.tolist(), "v": vf.tolist()})

    def calc_initialvalue(self, solver, bias):
        center = (solver.pde.domain[0] + solver.pde.domain[1]) / 2
        u0 = solver.space.function()
        p = solver.space.interpolation_points()
        u0[:] = solver.pde.initial_value(p, center=center + bias)
        self.export_data(uf=u0)

    def calc_gradientflow(self, solver, maxit=10000, tol=1e-11):
        uf = solver.space.function()
        u0 = self.input_data()
        uf[:] = solver.calculate_gradientflow(u0, maxit=maxit, tol=tol)
        self.export_data(uf)

    def calc_saddle(self, solver, maxit=10000, tol=1e-11, idx=1, v0=None):
        u0 = self.input_data()
        if v0 is not None:
            v0 = solver.space.function()
            idata = loadmat(self.ipath)
            v0 = idata["v"]
        uf, vf = solver.calculate_saddle(u0, tol=tol, maxit=maxit, idx=idx, v0=v0)
        self.export_data(uf=uf, vf=vf)


if __name__ == "__main__":
    box = (0, 1.6, 0, 0.7)
    bottom_gap = 0.06
    shape_parameter = (0.045, 0.14)
    domain = DomainRectangularBottom(
        box=box, shape_parameter=shape_parameter, bottom_gap=bottom_gap
    )

    pde = DropletLattice(domain=domain, mesh_file="./mesh_data/test2_heightdot14.vol")
    space = LagrangeFiniteElementSpace(pde.mesh, 1)
    p = space.interpolation_points()
    solver = ChisdAllenCahn(pde)

    test = Test(ipath="./solution_data/0000_UUUD.mat")

    # test.calc_initialvalue(solver, bias=-0.03)

    # find gradient flow
    test.calc_gradientflow(solver=solver, maxit=50000)

    # find saddle
    #test.calc_saddle(solver=solver, idx=1, tol=1e-11, maxit=50000)

    # test.calc_saddle(solver=solver, idx=1, tol = 1e-11, maxit = 50000, v0="")
