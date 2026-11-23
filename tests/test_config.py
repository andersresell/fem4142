"""Checks that invalid problem settings give clear errors."""
import pytest

from fem.useful_imports import *


def test_invalid_element_type_is_rejected():
    with pytest.raises(ValueError, match="ElementType"):
        create_config(1.0, 0.3, 1.0, 99, ProblemType.PLANE_STRESS)


def test_invalid_problem_type_is_rejected():
    with pytest.raises(ValueError, match="ProblemType"):
        create_config(1.0, 0.3, 1.0, ElementType.Q4, "plane stress")


@pytest.mark.parametrize("problem_type, dof", [(ProblemType.PLANE_STRESS, Dof.W), (ProblemType.PLATE, Dof.U),
                                               (ProblemType.PLANE_STRESS, 99)],
                         ids=["W in plane stress", "U in plate", "not a dof"])
def test_invalid_dof_is_rejected(problem_type, dof):
    config = create_config(1.0, 0.3, 1.0, ElementType.Q4, problem_type)
    mesh = create_structured_quad_mesh(config, 1.0, 1.0, 1, 1)
    with pytest.raises(Exception, match="Dof"):
        add_boundary_condition(config, mesh, "west", dof, 0)


def test_invalid_load_type_is_rejected():
    config = create_config(1.0, 0.3, 1.0, ElementType.Q4, ProblemType.PLANE_STRESS)
    mesh = create_structured_quad_mesh(config, 1.0, 1.0, 1, 1)
    with pytest.raises(ValueError, match="LoadType"):
        add_load(config, mesh, "north", 99, lambda x, y: 1.0)
