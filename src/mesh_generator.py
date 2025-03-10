import math
import os
from math import pi

import netgen.geom2d as g2d
import numpy as np
from fealpy.mesh import TriangleMesh


# from ngsolve import *
class DomainRectangular:
    def __init__(self, box=(0.0, 1.0, 0.0, 1.0)):
        self.box = box
        self.Lx = self.box[1] - self.box[0]
        self.Ly = self.box[3] - self.box[0]

    def geometry_info(self):
        geo = g2d.SplineGeometry()
        p_x = [self.box[0], self.box[0], self.box[1], self.box[1]]
        p_y = [self.box[3], self.box[2], self.box[2], self.box[3]]
        nn = len(p_x)
        lines = []
        for i in range(nn):
            lines.append((i, (i + 1) % nn))
        geo_nodes = [geo.AppendPoint(*p) for p in zip(p_x, p_y)]
        for p1, p2 in lines:
            geo.Append(["line", geo_nodes[p1], geo_nodes[p2]])
        return geo


class DomainRectangularBottom(DomainRectangular):
    # shape Parameter: (bottomWidth, bottomHeight)
    def __init__(
        self, box=(0, 1.6, 0, 0.7), shape_parameter=(0.045, 0.1), bottom_gap=0.06
    ):
        super(DomainRectangularBottom, self).__init__(box=box)
        self.shape_parameter = shape_parameter
        self.bottom_gap = bottom_gap

    def generate_unit(self, start_coordinate):
        width = self.shape_parameter[0]
        height = self.shape_parameter[1]
        unit_ref_x = [0, 0, width, width]
        unit_ref_y = [0, height, height, 0]
        p_x = [ele + start_coordinate[0] for ele in unit_ref_x]
        p_y = unit_ref_y.copy()
        return p_x, p_y

    def geometry_info(self):
        geo = g2d.SplineGeometry()

        prec_x = [self.box[0], self.box[0], self.box[1], self.box[1]]
        prec_y = [self.box[3], self.box[2], self.box[2], self.box[3]]

        width = self.shape_parameter[0]
        num_units = math.floor((self.Lx + self.bottom_gap) // (width + self.bottom_gap))
        occupy_width = num_units * width + (num_units - 1) * self.bottom_gap

        assert occupy_width < self.Lx

        leftover = (self.Lx - occupy_width) / 2

        px = [None] * (4 * (num_units + 1))
        py = [None] * (4 * (num_units + 1))
        px[0:2] = prec_x[0:2]
        px[-2:] = prec_x[-2:]
        py[0:2] = prec_y[0:2]
        py[-2:] = prec_y[-2:]

        for ii in range(0, num_units):
            start_x = leftover + ii * (self.shape_parameter[0] + self.bottom_gap)
            unit_x, unit_y = self.generate_unit((start_x, 0))
            px[(2 + ii * 4) : (6 + ii * 4)] = unit_x
            py[(2 + ii * 4) : (6 + ii * 4)] = unit_y

        nn = len(px)
        lines = []
        for i in range(nn):
            lines.append((i, (i + 1) % nn))
        geo_nodes = [geo.AppendPoint(*p) for p in zip(px, py)]
        for p1, p2 in lines:
            geo.Append(["line", geo_nodes[p1], geo_nodes[p2]])
        return geo


class DomainParallelogramBottom(DomainRectangular):
    # shape Parameter: (bottomWidth, bottomHeight, helixAngle)
    def __init__(
        self, box=(0, 1.6, 0, 0.7), shape_parameter=(0.045, 0.1, 40), bottom_gap=0.06
    ):
        super(DomainParallelogramBottom, self).__init__(box=box)
        self.shape_parameter = shape_parameter
        self.bottom_gap = bottom_gap

    def generate_unit(self, start_coordinate):
        width = self.shape_parameter[0]
        height = self.shape_parameter[1]
        arc = self.shape_parameter[2] * pi / 180
        unit_ref_x = [0, height / math.tan(arc), height / math.tan(arc) + width, width]
        unit_ref_y = [0, height, height, 0]
        p_x = [ele + start_coordinate[0] for ele in unit_ref_x]
        p_y = unit_ref_y.copy()
        return p_x, p_y

    def geometry_info(self):
        geo = g2d.SplineGeometry()

        p_rec_x = [self.box[0], self.box[0], self.box[1], self.box[1]]
        p_rec_y = [self.box[3], self.box[2], self.box[2], self.box[3]]

        width = self.shape_parameter[0]
        height = self.shape_parameter[1]
        arc = self.shape_parameter[2] * pi / 180
        num_units = int(
            (self.Lx + self.bottom_gap - height / math.tan(arc))
            // (width + self.bottom_gap)
        )
        occupy_width = (
            num_units * width
            + (num_units - 1) * self.bottom_gap
            + height / math.tan(arc)
        )

        assert occupy_width < self.Lx

        leftover = (self.Lx - occupy_width) / 2

        p_x = [None] * (4 * (num_units + 1))
        p_y = [None] * (4 * (num_units + 1))
        p_x[0:2] = p_rec_x[0:2]
        p_x[-2:] = p_rec_x[-2:]
        p_y[0:2] = p_rec_y[0:2]
        p_y[-2:] = p_rec_y[-2:]

        for ii in range(0, num_units):
            start_x = leftover + ii * (self.shape_parameter[0] + self.bottom_gap)
            unit_x, unit_y = self.generate_unit((start_x, 0))
            p_x[(2 + ii * 4) : (6 + ii * 4)] = unit_x
            p_y[(2 + ii * 4) : (6 + ii * 4)] = unit_y

        nn = len(p_x)
        lines = []
        for i in range(nn):
            lines.append((i, (i + 1) % nn))
        geo_nodes = [geo.AppendPoint(*p) for p in zip(p_x, p_y)]
        for p1, p2 in lines:
            geo.Append(["line", geo_nodes[p1], geo_nodes[p2]])
        return geo


def generate_mesh(geo_info=None, maxh=0.5, mesh_file=None):
    geo_mesh = geo_info.GenerateMesh(maxh=maxh)
    grid_size = str(maxh)  # Convert maxh to string for naming the file

    if mesh_file is None:
        file_name = "lattice_" + grid_size + ".vol"  # Construct the file name
        file_path = os.path.join("./mesh_data/", file_name)  # Construct the file path
    else:
        file_path = mesh_file

    if not os.path.exists(file_path):
        geo_mesh.Save(file_path)
        print(f"Mesh saved to {file_path}")
    else:
        print(f"Mesh file {file_path} already exists, skipping save.")

    # read mesh and transform the data structure to those used in fealpy
    with open(file_path) as f:
        cell = []
        node = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                line = next(f)
                if "surfaceelements" in line:
                    n_next = int(next(f))
                    for i in range(n_next):
                        cell.append(next(f).split()[-3:])
                elif "points" in line:
                    n_next = int(next(f))
                    for i in range(n_next):
                        node.append(next(f).split()[0:-1])

    cell = np.array(cell).astype(np.int64) - 1
    node = np.array(node).astype(np.float64)
    mesh = TriangleMesh(node, cell)
    return mesh
