"""Checks of the stress recovery (post-processing)."""
import pytest

import fem.stress_recovery as stress_recovery
from fem.useful_imports import *

SAMPLE_POINTS = [(-1.0, -1.0), (1.0, 1.0), (0.3, -0.7), (0.0, 0.0)]


@pytest.mark.parametrize(
    "element_type, displacement_x, exact_sigma_xx",
    [
        # Q4 can represent u = x*y exactly, which gives sigma_xx = E * du/dx = y (E = 1, nu = 0)
        (ELEMENT_TYPE_Q4, lambda x, y: x * y, lambda x, y: y),
        # The quadratic and cubic elements can represent u = x^2 exactly, which gives sigma_xx = 2x
        (ELEMENT_TYPE_Q8, lambda x, y: x**2, lambda x, y: 2 * x),
        (ELEMENT_TYPE_Q9, lambda x, y: x**2, lambda x, y: 2 * x),
        (ELEMENT_TYPE_Q16, lambda x, y: x**2, lambda x, y: 2 * x),
    ],
    ids=["Q4", "Q8", "Q9", "Q16"])
def test_consistent_stress_is_exact_when_the_displacement_field_is_exact(element_type, displacement_x,
                                                                         exact_sigma_xx):
    config = create_config(1.0, 0.0, 1.0, element_type, PROBLEM_TYPE_PLANE_STRESS)
    mesh = create_structured_quad_mesh(config, 2.0, 2.0, 1, 1)  #One element covering [0, 2] x [0, 2]
    x, y = mesh.nodes[:, 0], mesh.nodes[:, 1]
    u = displacement_x(x, y)
    v = np.zeros_like(u)
    sigma = stress_recovery.recover_stress_plane_stress_consistent(0, config, mesh, mesh.nodes, u, v, SAMPLE_POINTS)
    for (xi, eta), sigma_point in zip(SAMPLE_POINTS, sigma):
        x_point, y_point = xi + 1, eta + 1  #Maps the reference element [-1, 1]^2 to [0, 2]^2
        assert sigma_point[0] == pytest.approx(exact_sigma_xx(x_point, y_point), abs=1e-12)
        assert sigma_point[1] == pytest.approx(0.0, abs=1e-12)
