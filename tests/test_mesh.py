"""Checks of the structured mesh generator."""
import pytest
from problems import ALL_ELEMENT_TYPES, element_name

import fem.shape_functions as shape_functions
from fem.useful_imports import *


def make_mesh(element_type, nEx=3, nEy=2, Lx=3.0, Ly=2.0, perturb_mesh_nodes_factor=0.0):
    config = create_config(1.0, 0.3, 1.0, element_type, ProblemType.PLANE_STRESS)
    return create_structured_quad_mesh(config, Lx, Ly, nEx, nEy, perturb_mesh_nodes_factor)


@pytest.mark.parametrize("element_type", ALL_ELEMENT_TYPES, ids=element_name)
def test_number_of_nodes_and_elements(element_type):
    nEx, nEy = 3, 2
    mesh = make_mesh(element_type, nEx, nEy)
    n = ELEMENT_TRAITS[element_type].nNl_1D - 1  # Number of node spacings along an element edge
    if ELEMENT_TRAITS[element_type].category == ElementCategory.LAGRANGIAN:
        expected_nN = (n * nEx + 1) * (n * nEy + 1)
    else:
        expected_nN = (2 * nEx + 1) * (2 * nEy + 1) - nEx * nEy  #Serendipity elements have no center node
    assert mesh.get_nN() == expected_nN
    assert mesh.get_nE() == nEx * nEy


@pytest.mark.parametrize("element_type", ALL_ELEMENT_TYPES, ids=element_name)
def test_neighbouring_elements_share_edge_nodes(element_type):
    """If edge nodes were created twice, we would get several nodes at the same position."""
    nodes = make_mesh(element_type).nodes
    distances = np.linalg.norm(nodes[:, None, :] - nodes[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    assert np.min(distances) > 1e-8


@pytest.mark.parametrize("element_type", ALL_ELEMENT_TYPES, ids=element_name)
def test_edge_node_sets_lie_on_their_edges(element_type):
    Lx, Ly, nEx, nEy = 3.0, 2.0, 3, 2
    mesh = make_mesh(element_type, nEx, nEy, Lx, Ly)
    n = ELEMENT_TRAITS[element_type].nNl_1D - 1
    x, y = mesh.nodes[:, 0], mesh.nodes[:, 1]
    assert np.allclose(x[mesh.node_sets["west"]], 0.0) and len(mesh.node_sets["west"]) == n * nEy + 1
    assert np.allclose(x[mesh.node_sets["east"]], Lx) and len(mesh.node_sets["east"]) == n * nEy + 1
    assert np.allclose(y[mesh.node_sets["south"]], 0.0) and len(mesh.node_sets["south"]) == n * nEx + 1
    assert np.allclose(y[mesh.node_sets["north"]], Ly) and len(mesh.node_sets["north"]) == n * nEx + 1


@pytest.mark.parametrize("element_type", ALL_ELEMENT_TYPES, ids=element_name)
def test_perturbed_mesh_keeps_its_outer_boundary_and_positive_jacobians(element_type):
    Lx, Ly = 3.0, 2.0
    mesh = make_mesh(element_type, 4, 4, Lx, Ly, perturb_mesh_nodes_factor=1.0)
    x, y = mesh.nodes[:, 0], mesh.nodes[:, 1]
    assert np.allclose(x[mesh.node_sets["west"]], 0.0) and np.allclose(x[mesh.node_sets["east"]], Lx)
    assert np.allclose(y[mesh.node_sets["south"]], 0.0) and np.allclose(y[mesh.node_sets["north"]], Ly)
    for e in range(mesh.get_nE()):
        x_l = mesh.nodes[mesh.elements[e, :], 0]
        y_l = mesh.nodes[mesh.elements[e, :], 1]
        for xi in [-1.0, 0.0, 1.0]:
            for eta in [-1.0, 0.0, 1.0]:
                J = shape_functions.calc_J(xi, eta, x_l, y_l, element_type)  #Fails internally if det(J) <= 0
                assert np.linalg.det(J) > 0


@pytest.mark.parametrize("nEx, nEy, Lx, Ly", [(0, 1, 1.0, 1.0), (1, 0, 1.0, 1.0), (1, 1, 0.0, 1.0), (1, 1, 1.0, -1.0)])
def test_invalid_mesh_input_raises_an_error(nEx, nEy, Lx, Ly):
    with pytest.raises(ValueError):
        make_mesh(ElementType.Q4, nEx, nEy, Lx, Ly)


@pytest.mark.parametrize("factor", [-0.1, 1.5])
def test_invalid_perturbation_factor_raises_an_error(factor):
    with pytest.raises(ValueError):
        make_mesh(ElementType.Q4, perturb_mesh_nodes_factor=factor)


def test_too_large_perturbation_gives_a_warning_about_invalid_elements(capsys):
    """With perturb_mesh_nodes_factor close to 1, a few elements can become non-convex, so that det(J) <= 0 near a
    corner. This is the mesh from main.py, where 2 of the 2000 elements become invalid."""
    make_mesh(ElementType.Q4, 200, 10, 10.0, 1.25, perturb_mesh_nodes_factor=1.0)
    output = capsys.readouterr().out
    assert "WARNING" in output and "[759, 1067]" in output


def test_moderate_perturbation_gives_no_warning(capsys):
    make_mesh(ElementType.Q4, 200, 10, 10.0, 1.25, perturb_mesh_nodes_factor=0.5)
    assert "WARNING" not in capsys.readouterr().out
