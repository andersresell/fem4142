"""Checks of the consistent load integration and the validation of user-supplied load functions."""
import math

import pytest
from problems import ALL_ELEMENT_TYPES, element_name

import fem.loads as loads
from fem.useful_imports import *

Lx, Ly, h = 4.0, 2.0, 0.5


def assemble_external_forces(problem_type, element_type, element_set_name, load_type, load_function):
    """Returns the sum of the nodal forces for each dof component, e.g. (Fx, Fy) for plane stress."""
    config = create_config(1.0, 0.3, h, element_type, problem_type)
    mesh = create_structured_quad_mesh(config, Lx, Ly, 3, 2, perturb_mesh_nodes_factor=0.5)
    add_load(config, mesh, element_set_name, load_type, load_function)
    solver_data = create_solver_data(config, mesh)
    loads.assemble_loads_consistent(config, mesh, solver_data)
    return [np.sum(component) for component in unpack_solution(config, solver_data.R_ext)]


@pytest.mark.parametrize("element_type", ALL_ELEMENT_TYPES, ids=element_name)
def test_total_force_from_uniform_pressure_on_top_edge(element_type):
    p = 10.0
    Fx, Fy = assemble_external_forces(ProblemType.PLANE_STRESS, element_type, "north", LoadType.PRESSURE,
                                      lambda x, y: p)
    assert np.isclose(Fx, 0.0)
    assert np.isclose(Fy, -p * Lx * h)  #A positive pressure pushes into the body (downwards on the top edge)


@pytest.mark.parametrize("element_type", ALL_ELEMENT_TYPES, ids=element_name)
def test_total_force_from_linearly_varying_traction(element_type):
    t0 = 10.0
    Fx, Fy = assemble_external_forces(ProblemType.PLANE_STRESS, element_type, "east", LoadType.TRACTION,
                                      lambda x, y: (t0 * y / Ly, 0.0))
    assert np.isclose(Fx, 0.5 * t0 * Ly * h)
    assert np.isclose(Fy, 0.0)


@pytest.mark.parametrize("element_type", ALL_ELEMENT_TYPES, ids=element_name)
def test_total_force_from_body_force(element_type):
    bx, by = 2.0, -3.0
    Fx, Fy = assemble_external_forces(ProblemType.PLANE_STRESS, element_type, "domain", LoadType.BODY_FORCE,
                                      lambda x, y: (bx, by))
    assert np.isclose(Fx, bx * Lx * Ly * h)
    assert np.isclose(Fy, by * Lx * Ly * h)


@pytest.mark.parametrize("element_type", ALL_ELEMENT_TYPES, ids=element_name)
def test_total_force_from_pressure_on_plate(element_type):
    q = 10.0
    Fz, Mx, My = assemble_external_forces(ProblemType.PLATE, element_type, "domain", LoadType.PRESSURE,
                                          lambda x, y: q)
    assert np.isclose(Fz, -q * Lx * Ly)
    assert Mx == 0.0 and My == 0.0


@pytest.mark.parametrize(
    "load_function",
    [
        lambda x, y: (1.0, 2.0),  # tuple
        lambda x, y: [1.0, 2.0],  # list
        lambda x, y: np.array([1.0, 2.0]),  # numpy array
        lambda x, y: (np.float64(1.0), np.int64(2)),  # numpy numbers
    ],
    ids=["tuple", "list", "numpy array", "numpy numbers"])
def test_vector_load_functions_of_common_types_are_accepted(load_function):
    Fx, Fy = assemble_external_forces(ProblemType.PLANE_STRESS, ElementType.Q4, "east", LoadType.TRACTION,
                                      load_function)
    assert np.isclose(Fx, 1.0 * Ly * h) and np.isclose(Fy, 2.0 * Ly * h)


@pytest.mark.parametrize("load_function", [lambda x, y: 3, lambda x, y: 3.0, lambda x, y: np.float64(3.0)],
                         ids=["int", "float", "numpy float"])
def test_scalar_load_functions_of_common_types_are_accepted(load_function):
    Fx, Fy = assemble_external_forces(ProblemType.PLANE_STRESS, ElementType.Q4, "north", LoadType.PRESSURE,
                                      load_function)
    assert np.isclose(Fy, -3.0 * Lx * h)


@pytest.mark.parametrize("load", [(1.0, 2.0), [1, 2], np.array([1.0, 2.0])], ids=["tuple", "list", "numpy array"])
def test_constant_vector_loads_are_accepted(load):
    Fx, Fy = assemble_external_forces(ProblemType.PLANE_STRESS, ElementType.Q9, "east", LoadType.TRACTION, load)
    assert np.isclose(Fx, 1.0 * Ly * h) and np.isclose(Fy, 2.0 * Ly * h)


@pytest.mark.parametrize("problem_type", [ProblemType.PLANE_STRESS, ProblemType.PLATE], ids=lambda p: p.name)
@pytest.mark.parametrize("load", [3, 3.0, np.float64(3.0)], ids=["int", "float", "numpy float"])
def test_constant_scalar_loads_give_the_same_forces_as_a_function(problem_type, load):
    if problem_type == ProblemType.PLANE_STRESS:
        element_set_name = "north"
    else:
        assert problem_type == ProblemType.PLATE
        element_set_name = "domain"
    from_constant = assemble_external_forces(problem_type, ElementType.Q9, element_set_name, LoadType.PRESSURE, load)
    from_function = assemble_external_forces(problem_type, ElementType.Q9, element_set_name, LoadType.PRESSURE,
                                             lambda x, y: 3.0)
    assert np.allclose(from_constant, from_function)


@pytest.mark.parametrize(
    "load_type, load_function",
    [
        (LoadType.TRACTION, lambda x, y: (1.0, 2.0, 3.0)),  # three components in 2D
        (LoadType.TRACTION, lambda x: (1.0, 2.0)),  # only one argument
        (LoadType.TRACTION, lambda x, y: 1.0),  # traction must be a vector
        (LoadType.PRESSURE, lambda x, y: (1.0, 2.0)),  # pressure must be a scalar
        (LoadType.TRACTION, 5.0),  # constant traction with one component
        (LoadType.PRESSURE, (1.0, 2.0)),  # constant pressure with two components
        (LoadType.PRESSURE, "10"),  # neither a number nor a function
        (LoadType.PRESSURE, None),  # no load
    ],
    ids=["three components", "one argument", "scalar traction", "vector pressure", "scalar constant traction",
         "vector constant pressure", "text", "None"])
def test_invalid_load_functions_are_rejected(load_type, load_function):
    config = create_config(1.0, 0.3, h, ElementType.Q4, ProblemType.PLANE_STRESS)
    mesh = create_structured_quad_mesh(config, Lx, Ly, 2, 2)
    if load_type == LoadType.TRACTION:
        element_set_name = "east"
    else:
        element_set_name = "north"
    with pytest.raises(ValueError):
        add_load(config, mesh, element_set_name, load_type, load_function)


def test_load_function_that_is_undefined_at_the_origin_is_accepted_when_the_origin_is_outside_the_mesh():
    """A quarter ring doesn't contain the origin, so a load like 1/r must be accepted"""
    config = create_config(1.0, 0.3, 1.0, ElementType.Q9, ProblemType.PLANE_STRESS)
    mesh = create_structured_quad_mesh(config, 1.0, 1.0, 2, 2)
    morph_mesh_to_quarter_annulus(mesh, inner_radius=1.0, outer_radius=2.0)
    add_load(config, mesh, "west", LoadType.PRESSURE, lambda x, y: 1 / math.sqrt(x**2 + y**2))
    assert len(config.loads) == 1
