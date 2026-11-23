import numpy as np
from fem.mesh import Mesh

#====================================================================
# Functions that move the nodes of a rectangular mesh (from
# create_structured_quad_mesh), so that the mesh gets another shape.
# The node sets and element sets (e.g. "west") follow the nodes.
# For other shapes, you can move the nodes yourself by changing
# mesh.nodes.
#====================================================================


def get_unit_square_coordinates(mesh: Mesh):
    """Returns the node coordinates (s, t) scaled to a unit square, so that the size of the rectangle doesn't matter."""
    x, y = mesh.nodes[:, 0], mesh.nodes[:, 1]
    s = (x - np.min(x)) / (np.max(x) - np.min(x))
    t = (y - np.min(y)) / (np.max(y) - np.min(y))
    return s, t


def morph_mesh_to_quarter_circle(mesh: Mesh, radius=1.0):
    """A quarter of a circle, centered at the origin. "south" lies along the x-axis and "west" along the y-axis.
    "east" and "north" each cover half of the arc."""
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}")
    s, t = get_unit_square_coordinates(mesh)
    #Maps the unit square to a quarter of the unit circle (the corner (1, 1) ends up in the middle of the arc)
    mesh.nodes[:, 0] = radius * s * np.sqrt(1 - t**2 / 2)
    mesh.nodes[:, 1] = radius * t * np.sqrt(1 - s**2 / 2)


def morph_mesh_to_quarter_annulus(mesh: Mesh, inner_radius=0.5, outer_radius=1.0):
    """A quarter of a ring, centered at the origin. "west" is the inner arc and "east" the outer arc.
    "south" lies along the x-axis and "north" along the y-axis."""
    if not 0 < inner_radius < outer_radius:
        raise ValueError(f"Must have 0 < inner_radius < outer_radius, got inner_radius={inner_radius} "
                         f"and outer_radius={outer_radius}")
    s, t = get_unit_square_coordinates(mesh)
    #Polar coordinates: s goes from the inner to the outer radius, and t goes from 0 to 90 degrees
    r = inner_radius + s * (outer_radius - inner_radius)
    angle = t * np.pi / 2
    mesh.nodes[:, 0] = r * np.cos(angle)
    mesh.nodes[:, 1] = r * np.sin(angle)


def morph_mesh_to_quadrilateral(mesh: Mesh, corners=((0, 0), (1, 0), (1, 1), (0, 1))):
    """A four-sided shape with straight edges. corners is a list of four points (x, y):
    South-west, south-east, north-east and north-west, i.e. counter-clockwise."""
    if np.shape(corners) != (4, 2):
        raise ValueError(f"corners must be a list of four points (x, y), got {corners}")
    corners = np.array(corners, dtype=float)
    #Walking counter-clockwise around a convex shape, we always turn left
    for k in range(4):
        a = corners[(k + 1) % 4] - corners[k]
        b = corners[(k + 2) % 4] - corners[(k + 1) % 4]
        if a[0] * b[1] - a[1] * b[0] <= 0:
            raise ValueError(f"The corners {corners.tolist()} must form a convex shape, and be ordered "
                             f"counter-clockwise: south-west, south-east, north-east, north-west")

    s, t = get_unit_square_coordinates(mesh)
    #Interpolate between the corners with the shape functions of a Q4 element
    N = [(1 - s) * (1 - t), s * (1 - t), s * t, (1 - s) * t]
    mesh.nodes[:, 0] = sum(N[i] * corners[i, 0] for i in range(4))
    mesh.nodes[:, 1] = sum(N[i] * corners[i, 1] for i in range(4))
