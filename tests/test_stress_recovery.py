"""Checks of the stress recovery (post-processing): The consistent stress and the L2 projection."""
import pytest
from problems import find_node_closest_to, solve_cantilever, solve_square_plate

import fem.shape_functions as shape_functions
import fem.stress_recovery as stress_recovery
from fem.useful_imports import *

SAMPLE_POINTS = [(-1.0, -1.0), (1.0, 1.0), (0.3, -0.7), (0.0, 0.0)]


def one_element_mesh(element_type, problem_type, E=1.0, nu=0.0, h=1.0):
    config = create_config(E, nu, h, element_type, problem_type)
    mesh = create_structured_quad_mesh(config, 2.0, 2.0, 1, 1)  #One element covering [0, 2] x [0, 2]
    return config, mesh


#====================================================================
# Consistent stress, for displacement fields that the elements can
# represent exactly
#====================================================================


@pytest.mark.parametrize(
    "element_type, displacement_x, exact_sigma_xx",
    [
        # Q4 can represent u = x*y exactly, which gives sigma_xx = E * du/dx = y (E = 1, nu = 0)
        (ElementType.Q4, lambda x, y: x * y, lambda x, y: y),
        # The quadratic and cubic elements can represent u = x^2 exactly, which gives sigma_xx = 2x
        (ElementType.Q8, lambda x, y: x**2, lambda x, y: 2 * x),
        (ElementType.Q9, lambda x, y: x**2, lambda x, y: 2 * x),
        (ElementType.Q16, lambda x, y: x**2, lambda x, y: 2 * x),
    ],
    ids=["Q4", "Q8", "Q9", "Q16"])
def test_consistent_stress_is_exact_when_the_displacement_field_is_exact(element_type, displacement_x,
                                                                         exact_sigma_xx):
    config, mesh = one_element_mesh(element_type, ProblemType.PLANE_STRESS)
    x, y = mesh.nodes[:, 0], mesh.nodes[:, 1]
    r = np.zeros(2 * mesh.get_nN())
    r[0::2] = displacement_x(x, y)
    sigma = stress_recovery.calc_stress_consistent(config, mesh, r, 0, SAMPLE_POINTS)
    for (xi, eta), sigma_point in zip(SAMPLE_POINTS, sigma):
        x_point, y_point = xi + 1, eta + 1  #Maps the reference element [-1, 1]^2 to [0, 2]^2
        assert sigma_point[0] == pytest.approx(exact_sigma_xx(x_point, y_point), abs=1e-12)
        assert sigma_point[1] == pytest.approx(0.0, abs=1e-12)


def test_plate_stress_resultants_are_exact_for_a_linear_field():
    """w = a*x, thetax = b*y and thetay = c*x. With the definitions in calc_Ke_mindlin_plate, the curvatures are
    kappa = (-dthetay/dx, dthetax/dy, dthetax/dx - dthetay/dy) = (-c, b, 0), and the shear strains are
    gamma = (dw/dx + thetay, dw/dy - thetax) = (a + c*x, -b*y)."""
    a, b, c = 0.3, -0.2, 0.5
    E, nu, h = 1.0, 0.25, 0.1
    config, mesh = one_element_mesh(ElementType.Q4, ProblemType.PLATE, E, nu, h)
    x, y = mesh.nodes[:, 0], mesh.nodes[:, 1]
    r = np.zeros(3 * mesh.get_nN())
    r[0::3] = a * x
    r[1::3] = b * y
    r[2::3] = c * x

    stress = stress_recovery.calc_stress_consistent(config, mesh, r, 0, SAMPLE_POINTS)
    D = E * h**3 / (12 * (1 - nu**2))
    shear_stiffness = 5 / 6 * h * E / (2 * (1 + nu))
    kappa_xx, kappa_yy = -c, b
    for (xi, eta), (M_xx, M_yy, M_xy, V_x, V_y) in zip(SAMPLE_POINTS, stress):
        x_point, y_point = xi + 1, eta + 1
        assert M_xx == pytest.approx(D * (kappa_xx + nu * kappa_yy))
        assert M_yy == pytest.approx(D * (nu * kappa_xx + kappa_yy))
        assert M_xy == pytest.approx(0.0, abs=1e-14)
        assert V_x == pytest.approx(shear_stiffness * (a + c * x_point))
        assert V_y == pytest.approx(shear_stiffness * (-b * y_point))


#====================================================================
# L2 projection
#====================================================================


def test_L2_projection_reproduces_a_stress_field_that_the_elements_can_represent():
    """u = x*y^2 gives sigma_xx = y^2 and sigma_xy = 0.5 * 2*x*y (E = 1, nu = 0), which Q9 can represent.
    The best fit of a field that can be represented exactly is the field itself."""
    config = create_config(1.0, 0.0, 1.0, ElementType.Q9, ProblemType.PLANE_STRESS)
    mesh = create_structured_quad_mesh(config, 2.0, 1.0, 3, 2)
    x, y = mesh.nodes[:, 0], mesh.nodes[:, 1]
    r = np.zeros(2 * mesh.get_nN())
    r[0::2] = x * y**2
    nodal_stress = stress_recovery.calc_nodal_stress_L2(config, mesh, r)
    assert np.allclose(nodal_stress[:, 0], y**2)
    assert np.allclose(nodal_stress[:, 1], 0.0)
    assert np.allclose(nodal_stress[:, 2], x * y)


#====================================================================
# Comparison with beam and plate theory
#====================================================================


@pytest.mark.parametrize("method", ["consistent", "L2"])
def test_cantilever_bending_stress_matches_beam_theory(method):
    """sigma_xx = M * c / I at the top of the beam, in the middle of the span"""
    Lx, Ly, h, P = 10.0, 1.0, 1.0, 1000.0
    config, mesh, solver_data = solve_cantilever(ElementType.Q9, 40, 4, Lx, Ly, h, tip_force=P)
    if method == "consistent":
        e = 3 * 40 + 20  #The element in the top row whose upper left corner is at (Lx/2, Ly)
        sigma_xx = stress_recovery.calc_stress_consistent(config, mesh, solver_data.r, e, [(-1.0, 1.0)])[0, 0]
    else:
        nodal_stress = stress_recovery.calc_nodal_stress_L2(config, mesh, solver_data.r)
        sigma_xx = nodal_stress[find_node_closest_to(mesh, Lx / 2, Ly), 0]
    I = h * Ly**3 / 12
    assert sigma_xx == pytest.approx(P * (Lx / 2) * (Ly / 2) / I, rel=0.005)


def test_simply_supported_plate_bending_moment_matches_kirchhoff_theory():
    L, q = 1.0, 1000.0
    config, mesh, solver_data = solve_square_plate(ElementType.Q16, 8, L=L, h=0.01, q=q)
    nodal_stress = stress_recovery.calc_nodal_stress_L2(config, mesh, solver_data.r)
    M_xx, M_yy, M_xy, V_x, V_y = nodal_stress[find_node_closest_to(mesh, L / 2, L / 2)]
    assert M_xx == pytest.approx(0.0479 * q * L**2, rel=0.02)  #Positive: tension at the bottom
    assert M_yy == pytest.approx(M_xx)  #Symmetry
    assert M_xy == pytest.approx(0.0, abs=1e-6 * M_xx)


def test_plate_shear_forces_carry_the_load_to_the_supports():
    """Vertical equilibrium: The shear forces along the edges must add up to the total load. A thick plate is
    used, since the shear forces of thin Mindlin plates converge slowly (a small error in the shear strain
    is multiplied by a very large shear stiffness)."""
    L, q = 1.0, 1000.0
    config, mesh, solver_data = solve_square_plate(ElementType.Q16, 8, L=L, h=0.1, q=q)
    xi_gauss, w_gauss = shape_functions.get_arr_xi(4), shape_functions.get_arr_w(4)
    shear_force_west = 0.0
    for e in mesh.element_sets["west"].elementIDs:
        points = [(-1.0, eta) for eta in xi_gauss]
        V_x = stress_recovery.calc_stress_consistent(config, mesh, solver_data.r, e, points)[:, 3]
        y_l = mesh.nodes[mesh.elements[e], 1]
        dy_deta = np.array([shape_functions.calc_dNdxi_dNdeta(-1.0, eta, config.element_type)[1] @ y_l
                            for eta in xi_gauss])
        shear_force_west += np.sum(-V_x * dy_deta * w_gauss)  #The outward normal of the west edge is -x
    assert 4 * shear_force_west == pytest.approx(q * L**2, rel=0.02)  #The four edges carry the same force
