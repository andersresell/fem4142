import numpy as np
from fem.mesh import Mesh

MORPH_TYPES = ["quarter_circle", "quarter_annulus", "quadrilateral"]


def morph_mesh(mesh: Mesh, morph_type: str, radius=1.0, inner_radius=0.5, corners=None):
    """Moves the nodes of a rectangular mesh (from create_structured_quad_mesh), so that the mesh gets
    another shape. The node sets and element sets (e.g. "west") follow the nodes. The morph types are:

        "quarter_circle":  A quarter of a circle with the given radius, centered at the origin.
                           "south" lies along the x-axis and "west" along the y-axis.
                           "east" and "north" each cover half of the arc.
        "quarter_annulus": A quarter of a ring between inner_radius and radius, centered at the origin.
                           "west" is the inner arc and "east" the outer arc.
                           "south" lies along the x-axis and "north" along the y-axis.
        "quadrilateral":   A four-sided shape with straight edges. corners is a list of four points (x, y):
                           South-west, south-east, north-east and north-west, i.e. counter-clockwise.

    Example: morph_mesh(mesh, "quarter_annulus", radius=2.0, inner_radius=1.0)

    For other shapes, you can move the nodes yourself by changing mesh.nodes.
    """
    #Scale the rectangle to a unit square with coordinates (s, t), so that the size of the original mesh doesn't matter
    x, y = mesh.nodes[:, 0], mesh.nodes[:, 1]
    s = (x - np.min(x)) / (np.max(x) - np.min(x))
    t = (y - np.min(y)) / (np.max(y) - np.min(y))

    if morph_type == "quarter_circle":
        if radius <= 0:
            raise ValueError(f"radius must be positive, got {radius}")
        #Maps the unit square to a quarter of the unit circle (the corner (1, 1) ends up in the middle of the arc)
        mesh.nodes[:, 0] = radius * s * np.sqrt(1 - t**2 / 2)
        mesh.nodes[:, 1] = radius * t * np.sqrt(1 - s**2 / 2)

    elif morph_type == "quarter_annulus":
        if not 0 < inner_radius < radius:
            raise ValueError(f"inner_radius must be between 0 and radius, got inner_radius={inner_radius} "
                             f"and radius={radius}")
        #Polar coordinates: s goes from the inner to the outer radius, and t goes from 0 to 90 degrees
        r = inner_radius + s * (radius - inner_radius)
        angle = t * np.pi / 2
        mesh.nodes[:, 0] = r * np.cos(angle)
        mesh.nodes[:, 1] = r * np.sin(angle)

    elif morph_type == "quadrilateral":
        corners = check_quadrilateral_corners(corners)
        #Interpolate between the corners with the shape functions of a Q4 element
        N = [(1 - s) * (1 - t), s * (1 - t), s * t, (1 - s) * t]
        mesh.nodes[:, 0] = sum(N[i] * corners[i, 0] for i in range(4))
        mesh.nodes[:, 1] = sum(N[i] * corners[i, 1] for i in range(4))

    else:
        raise ValueError(f"Unknown morph type '{morph_type}'. Available: {', '.join(MORPH_TYPES)}")


def check_quadrilateral_corners(corners):
    if corners is None or np.shape(corners) != (4, 2):
        raise ValueError(f"corners must be a list of four points (x, y), got {corners}")
    corners = np.array(corners, dtype=float)
    #Walking counter-clockwise around a convex shape, we always turn left
    for k in range(4):
        a = corners[(k + 1) % 4] - corners[k]
        b = corners[(k + 2) % 4] - corners[(k + 1) % 4]
        if a[0] * b[1] - a[1] * b[0] <= 0:
            raise ValueError(f"The corners {corners.tolist()} must form a convex shape, and be ordered "
                             f"counter-clockwise: south-west, south-east, north-east, north-west")
    return corners
