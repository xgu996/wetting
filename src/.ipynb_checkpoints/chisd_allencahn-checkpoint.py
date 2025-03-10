import random
import sys

sys.path.append(".")
sys.path.append("..")
import matplotlib.pyplot as plt
import numpy as np
from fealpy.functionspace import LagrangeFiniteElementSpace
from opt_einsum import contract
from scipy.io import loadmat, savemat
from scipy.linalg import eigh
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import LinearOperator, eigs, eigsh, spsolve

from model_data import DropletLattice


class ChisdAllenCahn(object):
    def __init__(self, pde, mesh=None):
        self.pde = pde
        self.mesh = self.pde.mesh
        self.space = LagrangeFiniteElementSpace(self.mesh, p=1)

        qf1 = self.mesh.integrator(8, "cell")
        qf2 = self.mesh.integrator(8, "edge")
        bcs1, ws1 = qf1.get_quadrature_points_and_weights()
        bcs2, ws2 = qf2.get_quadrature_points_and_weights()
        self.bcs = [bcs1, bcs2]
        self.ws = [ws1, ws2]
        self.phi = [self.space.basis(self.bcs[0]), self.space.face_basis(self.bcs[1])]

        # space variable
        self.S = self.space.stiff_matrix()
        self.M = self.space.mass_matrix()
        self.b = self.space.integral_basis()
        self.b0 = self.b.copy()
        self.b0[self.space.is_boundary_dof()] = 0

        idx = self.mesh.ds.boundary_edge_index()
        bc = self.mesh.entity_barycenter("edge", index=idx)
        flag = self.pde.is_robin_boundary(bc)

        self.idx = idx[flag]  # robin index
        self.cellmeasure = self.mesh.entity_measure("cell")
        self.edgemeasure = self.mesh.entity_measure("edge")[self.idx]
        self.measure = [self.cellmeasure, self.edgemeasure]
        self.geo2dof = [self.space.cell_to_dof(), self.space.edge_to_dof()[self.idx]]

        self.R = self.boundary_matrix()

    def fenergy(self, uh):
        pde = self.pde
        val = pde.epsilon * uh @ self.S @ uh / 2
        nlin = [
            pde.f(uh(self.bcs[0])) / pde.epsilon,
            pde.g(uh(self.bcs[1], index=self.idx)),
        ]
        for i in range(2):
            val += self.ws[i] @ nlin[i] @ self.measure[i]
        return val

    def grad_fenergy(self, uh):
        pde = self.pde
        nlin = [
            pde.diff_f(uh(self.bcs[0]), 1) / pde.epsilon,
            pde.diff_g(uh(self.bcs[1], index=self.idx), 1),
        ]
        gradVal = pde.epsilon * self.S @ uh
        for i in range(2):
            f = np.einsum(
                "i,j,ijk,ij->jk",
                self.ws[i],
                self.measure[i],
                self.phi[i],
                nlin[i],
                optimize="greedy",
            )
            np.add.at(gradVal, self.geo2dof[i], f)
        return gradVal

    def hess_fenergy(self, uh):
        pde = self.pde
        nlin = [
            pde.diff_f(uh(self.bcs[0]), 2) / pde.epsilon,
            pde.diff_g(uh(self.bcs[1], index=self.idx), 2),
        ]
        hessval = pde.epsilon * self.S
        gdof = self.space.number_of_global_dofs()
        for i in range(2):
            f = np.einsum(
                "i,j,ijk,ijl,ij->jkl",
                self.ws[i],
                self.measure[i],
                self.phi[i],
                self.phi[i],
                nlin[i],
                optimize="greedy",
            )
            I = np.broadcast_to(self.geo2dof[i][:, :, None], shape=f.shape)
            J = np.broadcast_to(self.geo2dof[i][:, None, :], shape=f.shape)
            hessval += csr_matrix((f.flat, (I.flat, J.flat)), shape=(gdof, gdof))
        return hessval

    def grad_nonlinear(self, uh, overwrite=None):
        if overwrite is None:
            gradval = self.space.function()
        else:
            gradval = overwrite
        pde = self.pde
        nlin = [
            pde.diff_f(uh(self.bcs[0]), 1) / pde.epsilon,
            pde.diff_g(uh(self.bcs[1], index=self.idx), 1),
        ]
        for i in range(2):
            f = np.einsum(
                "i,j,ijk,ij->jk",
                self.ws[i],
                self.measure[i],
                self.phi[i],
                nlin[i],
                optimize="greedy",
            )
            np.add.at(gradval, self.geo2dof[i], f)
        return gradval

    def calculate_eigs(self, H, n=2, which="SR"):
        def matVec(vec):
            vv = H @ (vec - self.b @ vec / np.sum(self.b))
            return vv - self.b * np.sum(vv) / np.sum(self.b)

        gdof = self.space.number_of_global_dofs()
        linop = LinearOperator((gdof, gdof), matvec=matVec)
        evals, evecs = eigs(A=linop, M=self.M, which="SR", k=n)
        idx = evals.real.argsort()
        evals.real.sort()
        evecs = evecs[:, idx].real
        return evals.real, evecs.real

    def calculate_gradientflow(self, uh, maxit=10000, tol=1e-10):
        space = self.space
        e0 = self.fenergy(uh)
        u = space.function()
        u[:] = uh
        dt = 5e-2 * self.pde.epsilon
        error = 1
        nit = 1
        while error > tol and nit <= maxit:
            mgrad = self.grad_fenergy(u)
            grad = spsolve(self.M, mgrad)
            rgrad = grad - self.b @ grad / np.sum(self.b)
            u -= dt * rgrad
            m1 = self.b @ u
            e1 = self.fenergy(u)
            error = np.linalg.norm(e1 - e0) / np.linalg.norm(e1)
            e0 = e1
            nit += 1
            print(nit)
            info = "fenery: {:.4e},  mass: {:.4e}, error: {:.4e}"
            print(info.format(e1, m1, error))
            print(np.linalg.norm(rgrad))
        return u

    def calculate_saddle(self, uh, maxit=10000, tol=1e-11, idx=1, v0=None):
        space = self.space
        gdof = space.number_of_global_dofs()
        unow = space.function()
        unext = space.function()
        vnow = np.zeros((gdof, idx), dtype=np.float64)
        vnext = np.zeros((gdof, idx), dtype=np.float64)

        unow[:] = uh

        Md = self.M.toarray()

        if v0 is None:
            # compute init eigenvalue
            H = self.hess_fenergy(unow)
            eval, evec = self.calculate_eigs(H, n=idx)
            print(eval)
            norm = contract("ij,kj,ik->j", evec, evec, Md)
            vnow = evec / np.sqrt(norm)
        else:
            vnow = v0.copy()

        dt = 5e-2 * self.pde.epsilon

        error = 1
        nit = 1

        print("--Initial energy--")
        e0 = self.fenergy(uh)
        print("Initial free energy \n", e0)
        print("Initial Mass\n", self.b @ unow)
        print("--  >>  <<  --")

        while error > tol and nit <= maxit:
            # comput next u
            mgrad = self.grad_fenergy(unow)  # modified grad
            grad = spsolve(self.M, mgrad)  # grad
            rgrad = grad - self.b @ grad / np.sum(self.b)  # riemann grad
            Mgrad = self.M @ rgrad
            du = rgrad - 2 * contract("ji,ki,k->j", vnow, vnow, Mgrad)
            unext[:] = unow - dt * du

            # hessian
            mhess = self.hess_fenergy(unext)

            for ii in range(1, idx + 1):
                vi = vnow[:, ii - 1]
                wi = spsolve(self.M, mhess @ vi)
                wi -= self.b @ wi / np.sum(self.b)
                Mwi = self.M @ wi
                vsi = vi - dt * (wi - (vi @ (Mwi) * vi))

                if ii >= 2:
                    vsi += (
                        2
                        * dt
                        * contract(
                            "il,jl,j->i", vnow[:, : (ii - 1)], vnow[:, : (ii - 1)], Mwi
                        )
                    )
                    Mvsi = self.M @ vsi
                    vsi -= contract(
                        "ij,i,lj->l",
                        vnext[:, : (ii - 1)],
                        Mvsi,
                        vnext[:, : (ii - 1)],
                    )
                    # check
                vnext[:, ii - 1] = vsi / np.sqrt(vsi @ (self.M @ vsi))

            unow[:] = unext
            vnow = vnext.copy()

            e1 = self.fenergy(unow)
            error = np.linalg.norm(e1 - e0) / np.linalg.norm(e1)
            info = "nit: {:d}, fenery: {:.8e},  energy_error: {:.4e} \n"
            print(info.format(nit, e1, error))
            print("norm grad val: {:.4e}".format(np.linalg.norm(rgrad)))
            print(self.b @ unow)
            nit += 1
            e0 = e1

            if (nit % 300) == 0:
                H = self.hess_fenergy(unow)
                evl, _ = self.calculate_eigs(H, n=10)
                print(evl)
        return unow, vnow

    def calculate_perturb(self, uh, a, direction):
        # direction (gdof, n)
        weight = 2 * np.random.rand(direction.shape[1]) - 1  # (n,)
        up = self.space.function()
        up[:] = uh + a * np.sum(direction * weight, axis=1)  # (gdof,)
        Hp = self.hess_fenergy(up)
        evalp, _ = self.calculate_eigs(Hp, 30)
        print("first 10 eigenvalues: \n")
        print(evalp)
        return up

    def boundary_matrix(self):
        r = np.einsum(
            "i,j,ijk,ijl->jkl", self.ws[1], self.measure[1], self.phi[1], self.phi[1]
        )
        edge2dof = self.space.edge_to_dof()[self.idx]
        gdof = self.space.number_of_global_dofs()
        I = np.broadcast_to(edge2dof[:, :, None], shape=r.shape)
        J = np.broadcast_to(edge2dof[:, None, :], shape=r.shape)
        R = csr_matrix((r.flat, (I.flat, J.flat)), shape=(gdof, gdof))
        return R


if __name__ == "__main__":
    # Set the Model
    pde = DropletLattice(theta=102, maxh=0.01)
    solver = ChisdAllenCahn(pde)
    space = solver.space
    p = space.interpolation_points()

    # # ## Set the initial value

    # uh = space.function()
    # uh[:] = solver.pde.initial_value(p)
    # plt.tricontourf(p[..., 0], p[..., 1], uh.tolist(), cmap="jet")

    # data = loadmat('./data_120_2/sd0_0.mat')
    # uh[:] = data['u'].flatten()
    # plt.tricontourf(p[..., 0], p[..., 1], uh.tolist(), cmap="jet")
    #
    # # # # --------------------------------------------------------------------------
    # # # # Solve the gradient flow
    # u1 = solver.compute_minimum(uh, tol=1e-11)
    # H1 = solver.hess_fenergy(u1)
    # eval1, evec1 = solver.comput_eigenvalues_and_eigenvectors(H1, 5)
    # print(eval1[:5])
    # plt.tricontourf(p[..., 0], p[..., 1], u1.tolist(), cmap="jet")

    # # --------------------------------------------------------------------------
    # # Perturb the solution

    gdof = space.number_of_global_dofs()

    # data = loadmat("./data_120_3/sd3.mat")
    # data = loadmat("./data_120_3/sd0_1_1.mat")
    data = loadmat("../test3/solution_data/000_UUU.mat")

    u = space.function()
    u[:] = data["u"].flatten()
    H = solver.hess_fenergy(u)
    start = 20
    end = 20
    eval, evec = solver.calculate_eigs(H, n=end)
    print(eval[0:3])
    # perturb_direction =evec[:, start:end]
    perturb_direction = np.hstack(
        (evec[:, 0:1].reshape(gdof, 1), evec[:, start:end])
    )  # (gdof, 3)
    up = solver.calculate_perturb(uh=u, a=0.5, direction=perturb_direction)
    plt.tricontourf(p[..., 0], p[..., 1], up.tolist(), cmap="jet")
    print(solver.b @ u)
    print(solver.b @ up)
