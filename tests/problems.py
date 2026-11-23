"""Small standard problems that are used by several of the test files."""
from fem.useful_imports import *

# All element types that are fully implemented (ELEMENT_TYPE_Q4_USER is left for students to implement)
ALL_ELEMENT_TYPES = [
    ELEMENT_TYPE_Q4, ELEMENT_TYPE_Q4R, ELEMENT_TYPE_Q8, ELEMENT_TYPE_Q8R, ELEMENT_TYPE_Q9, ELEMENT_TYPE_Q9R,
    ELEMENT_TYPE_Q16
]


def element_name(element_type):
    """Used to give readable test names, e.g. test_patch_test[Q9] instead of test_patch_test[4]"""
    return element_type_to_str[element_type]


def solve_cantilever(element_type, nEx, nEy, Lx=10.0, Ly=1.0, h=1.0, E=210e9, nu=0.3, tip_force=1000.0):
    """Plane stress cantilever clamped at the west edge. The east edge carries a uniform downward
    shear traction whose resultant is tip_force."""
    config = create_config(E, nu, h, element_type, PROBLEM_TYPE_PLANE_STRESS)
    mesh = create_structured_quad_mesh(config, Lx, Ly, nEx, nEy)
    add_boundary_condition(config, mesh, "west", DOF_U, 0)
    add_boundary_condition(config, mesh, "west", DOF_V, 0)
    traction_y = -tip_force / (Ly * h)
    add_load(config, mesh, "east", LOAD_TYPE_TRACTION, lambda x, y: (0.0, traction_y))
    solver_data = create_solver_data(config, mesh)
    solve(config, solver_data, mesh)
    return config, mesh, solver_data


def solve_square_plate(element_type, nE, L=1.0, h=0.01, E=200e9, nu=0.3, q=1000.0, clamped=False):
    """Square Mindlin plate (L x L) under uniform pressure q, with all four edges either simply
    supported (only w = 0) or clamped (w = thetax = thetay = 0). nE elements in each direction."""
    config = create_config(E, nu, h, element_type, PROBLEM_TYPE_PLATE)
    mesh = create_structured_quad_mesh(config, L, L, nE, nE)
    for edge in ["west", "east", "south", "north"]:
        add_boundary_condition(config, mesh, edge, DOF_W, 0)
        if clamped:
            add_boundary_condition(config, mesh, edge, DOF_THETAX, 0)
            add_boundary_condition(config, mesh, edge, DOF_THETAY, 0)
    add_load(config, mesh, "domain", LOAD_TYPE_PRESSURE, lambda x, y: q)
    solver_data = create_solver_data(config, mesh)
    solve(config, solver_data, mesh)
    return config, mesh, solver_data


def find_node_closest_to(mesh, x, y):
    return np.argmin((mesh.nodes[:, 0] - x)**2 + (mesh.nodes[:, 1] - y)**2)
