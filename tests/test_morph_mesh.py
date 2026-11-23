"""Checks of morph_mesh, which moves the nodes of a rectangular mesh into another shape."""
import pytest

import fem.shape_functions as shape_functions
from fem.useful_imports import *


def make_mesh(element_type=ElementType.Q16, nE=4):
    config = create_config(1.0, 0.3, 1.0, element_type, ProblemType.PLANE_STRESS)
    return create_structured_quad_mesh(config, 3.0, 2.0, nE, nE)  #The size doesn't matter, morph_mesh rescales it


def mesh_area(mesh, element_type):
    """Integrates 1 over the mesh. calc_J also checks that all elements have a positive Jacobian."""
    nGauss = ELEMENT_TRAITS[element_type].nNl_1D
    arr_xi, arr_w = shape_functions.get_arr_xi(nGauss), shape_functions.get_arr_w(nGauss)
    area = 0.0
    for e in range(mesh.get_nE()):
        x_l, y_l = mesh.nodes[mesh.elements[e, :], 0], mesh.nodes[mesh.elements[e, :], 1]
        for i in range(nGauss):
            for j in range(nGauss):
                J = shape_functions.calc_J(arr_xi[i], arr_xi[j], x_l, y_l, element_type)
                area += np.linalg.det(J) * arr_w[i] * arr_w[j]
    return area


def distance_from_origin(mesh, node_set_name):
    nodes = mesh.nodes[mesh.node_sets[node_set_name]]
    return np.sqrt(nodes[:, 0]**2 + nodes[:, 1]**2)


def test_quarter_circle():
    R = 2.0
    mesh = make_mesh()
    morph_mesh(mesh, "quarter_circle", radius=R)
    x, y = mesh.nodes[:, 0], mesh.nodes[:, 1]
    assert np.allclose(y[mesh.node_sets["south"]], 0.0) and np.allclose(x[mesh.node_sets["west"]], 0.0)
    assert np.allclose(distance_from_origin(mesh, "east"), R) and np.allclose(distance_from_origin(mesh, "north"), R)
    assert mesh_area(mesh, ElementType.Q16) == pytest.approx(np.pi * R**2 / 4, rel=1e-4)


def test_quarter_annulus():
    mesh = make_mesh()
    morph_mesh(mesh, "quarter_annulus", radius=2.0, inner_radius=1.0)
    x, y = mesh.nodes[:, 0], mesh.nodes[:, 1]
    assert np.allclose(distance_from_origin(mesh, "west"), 1.0) and np.allclose(distance_from_origin(mesh, "east"), 2.0)
    assert np.allclose(y[mesh.node_sets["south"]], 0.0) and np.allclose(x[mesh.node_sets["north"]], 0.0)
    assert mesh_area(mesh, ElementType.Q16) == pytest.approx(np.pi / 4 * (2.0**2 - 1.0**2), rel=1e-4)


def test_quadrilateral():
    corners = [(0.0, 0.0), (4.0, 0.0), (3.0, 2.0), (1.0, 3.0)]
    mesh = make_mesh(ElementType.Q4)
    morph_mesh(mesh, "quadrilateral", corners=corners)
    for corner_name, corner in zip(["south_west", "south_east", "north_east", "north_west"], corners):
        assert np.allclose(mesh.nodes[mesh.node_sets[corner_name][0]], corner)
    assert mesh_area(mesh, ElementType.Q4) == pytest.approx(7.5)  #Area of the quadrilateral


def test_simply_supported_circular_plate_matches_kirchhoff_theory():
    """A quarter of a thin circular plate, with symmetry conditions on the straight edges."""
    R, h, E, nu, q = 1.0, 0.01, 200e9, 0.3, 1000.0
    config = create_config(E, nu, h, ElementType.Q16, ProblemType.PLATE)
    mesh = create_structured_quad_mesh(config, R, R, 4, 4)
    morph_mesh(mesh, "quarter_circle", radius=R)
    add_boundary_condition(config, mesh, "west", Dof.THETAY, 0)
    add_boundary_condition(config, mesh, "south", Dof.THETAX, 0)
    add_boundary_condition(config, mesh, "east", Dof.W, 0)
    add_boundary_condition(config, mesh, "north", Dof.W, 0)
    add_load(config, mesh, "domain", LoadType.PRESSURE, lambda x, y: q)
    solver_data = create_solver_data(config, mesh)
    solve(config, solver_data, mesh)
    w, thetax, thetay = unpack_solution(config, solver_data.r)
    w_center = w[mesh.node_sets["south_west"][0]]

    D = E * h**3 / (12 * (1 - nu**2))
    w_center_kirchhoff = -(5 + nu) / (64 * (1 + nu)) * q * R**4 / D
    assert w_center == pytest.approx(w_center_kirchhoff, rel=0.01)


@pytest.mark.parametrize(
    "morph_type, parameters",
    [
        ("circle", {}),
        ("quarter_annulus", dict(radius=1.0, inner_radius=1.0)),
        ("quadrilateral", {}),
        ("quadrilateral", dict(corners=[(0, 0), (0, 1), (1, 1), (1, 0)])),
        ("quadrilateral", dict(corners=[(0, 0), (2, 0), (0.5, 0.5), (0, 2)])),
    ],
    ids=["unknown morph type", "inner radius too large", "no corners", "clockwise corners", "not convex"])
def test_invalid_morph_input_raises_an_error(morph_type, parameters):
    with pytest.raises(ValueError):
        morph_mesh(make_mesh(ElementType.Q4), morph_type, **parameters)
