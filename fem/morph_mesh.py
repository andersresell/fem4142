from fem.utils import Config
from fem.mesh import Mesh
import numpy as np

SMALL_VALUE = 1e-8


def morph_mesh(mesh: Mesh, morph_type: str):
    if morph_type == "quarter_circle_north_east_quadrant":
        nodes = mesh.nodes
        Lx = np.max(nodes[:, 0]) - np.min(nodes[:, 0])
        Ly = np.max(nodes[:, 1]) - np.min(nodes[:, 1])
        if not (np.isclose(Lx, 1) and np.isclose(Ly, 1)):
            raise ValueError("For quarter circle morph, mesh must be unit square in x and y directions")

        for I in range(nodes.shape[0]):

            x = nodes[I, 0]
            y = nodes[I, 1]
            # squircle?
            u = x * np.sqrt(1 - y * y / 2)
            v = y * np.sqrt(1 - x * x / 2)
            nodes[I, 0] = u
            nodes[I, 1] = v

    else:
        raise ValueError(f"Unknown morph type: {morph_type}")
