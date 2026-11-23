"""Checks of the consistent load integration and the validation of user-supplied load functions."""
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


@pytest.mark.parametrize(
    "load_type, load_function",
    [
        (LoadType.TRACTION, lambda x, y: (1.0, 2.0, 3.0)),  # three components in 2D
        (LoadType.TRACTION, lambda x: (1.0, 2.0)),  # only one argument
        (LoadType.TRACTION, lambda x, y: 1.0),  # traction must be a vector
        (LoadType.PRESSURE, lambda x, y: (1.0, 2.0)),  # pressure must be a scalar
        (LoadType.TRACTION, 5.0),  # not a function
    ],
    ids=["three components", "one argument", "scalar traction", "vector pressure", "not a function"])
def test_invalid_load_functions_are_rejected(load_type, load_function):
    config = create_config(1.0, 0.3, h, ElementType.Q4, ProblemType.PLANE_STRESS)
    mesh = create_structured_quad_mesh(config, Lx, Ly, 2, 2)
    if load_type == LoadType.TRACTION:
        element_set_name = "east"
    else:
        element_set_name = "north"
    with pytest.raises(ValueError):
        add_load(config, mesh, element_set_name, load_type, load_function)
