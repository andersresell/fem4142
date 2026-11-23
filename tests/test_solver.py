"""Checks of the complete solution procedure against exact and analytical solutions."""
import pytest
from problems import (ALL_ELEMENT_TYPES, element_name, find_node_closest_to, solve_cantilever, solve_square_plate)

from scipy.sparse import csc_matrix

import fem.element_stiffness as element_stiffness
import fem.stress_recovery as stress_recovery
from fem.solver import solve_linear_system
from fem.useful_imports import *

#====================================================================
# Patch test: A constant stress state must be reproduced exactly,
# also on distorted meshes
#====================================================================

# Q9R (Q9 with 2x2 Gauss points) has spurious zero energy modes that are not
# suppressed in a small patch, which makes the system singular. See the tests below.
PATCH_TEST_ELEMENT_TYPES = [et for et in ALL_ELEMENT_TYPES if et != ElementType.Q9R]


@pytest.mark.parametrize("element_type", PATCH_TEST_ELEMENT_TYPES, ids=element_name)
def test_patch_test_uniaxial_tension_on_distorted_mesh(element_type):
    """A plate pulled by the traction sigma0 on the east edge. The exact solution is
    u = sigma0 * x / E, v = -nu * sigma0 * y / E, and the stress is (sigma0, 0, 0) everywhere."""
    E, nu, sigma0 = 100.0, 0.25, 3.0
    config = create_config(E, nu, 1.0, element_type, ProblemType.PLANE_STRESS)
    mesh = create_structured_quad_mesh(config, 2.0, 1.0, 3, 3, perturb_mesh_nodes_factor=0.8)
    add_boundary_condition(config, mesh, "west", Dof.U, 0)
    add_boundary_condition(config, mesh, "south_west", Dof.V, 0)
    add_load(config, mesh, "east", LoadType.TRACTION, lambda x, y: (sigma0, 0.0))
    solver_data = create_solver_data(config, mesh)
    solve(config, solver_data, mesh)

    u, v = unpack_solution(config, solver_data.r)
    x, y = mesh.nodes[:, 0], mesh.nodes[:, 1]
    assert np.allclose(u, sigma0 * x / E, atol=1e-10)
    assert np.allclose(v, -nu * sigma0 * y / E, atol=1e-10)

    sample_points = [(-1.0, -1.0), (0.2, 0.7), (1.0, 0.0)]
    for e in range(mesh.get_nE()):
        sigma = stress_recovery.calc_stress_consistent(config, mesh, solver_data.r, e, sample_points)
        assert np.allclose(sigma, [sigma0, 0.0, 0.0], atol=1e-9)


def count_zero_energy_modes(Ke):
    eigenvalues = np.linalg.eigvalsh(Ke)
    return np.sum(eigenvalues < 1e-10 * np.max(eigenvalues))


def single_element_plane_stress_stiffness(element_type, hourglass_scaling=None):
    config = create_config(1.0, 0.3, 1.0, element_type, ProblemType.PLANE_STRESS)
    if hourglass_scaling is not None:
        config.hourglass_scaling = hourglass_scaling
    mesh = create_structured_quad_mesh(config, 1.0, 1.0, 1, 1)
    Ke = element_stiffness.calc_Ke_plane_stress(config, mesh, 0)
    if element_type == ElementType.Q4R:
        Ke += element_stiffness.calc_Ke_hourglass_plane_stress_Q4R(config, mesh, 0)
    return Ke


@pytest.mark.parametrize("element_type", [ElementType.Q4, ElementType.Q8, ElementType.Q9, ElementType.Q16],
                         ids=element_name)
def test_element_has_only_rigid_body_zero_energy_modes(element_type):
    """A single element in 2D has 3 rigid body modes (two translations and one rotation)."""
    Ke = single_element_plane_stress_stiffness(element_type)
    assert count_zero_energy_modes(Ke) == 3


def test_Q8R_has_one_spurious_zero_energy_mode():
    """With 2x2 Gauss points, Q8 has 1 spurious zero energy mode. It cannot spread between
    neighbouring elements, so it is usually harmless in a mesh."""
    assert count_zero_energy_modes(single_element_plane_stress_stiffness(ElementType.Q8R)) == 4


def test_Q4R_needs_hourglass_stabilization():
    """With one Gauss point, Q4 has 2 spurious zero energy (hourglass) modes in addition to
    the 3 rigid body modes. The hourglass stiffness removes them."""
    assert count_zero_energy_modes(single_element_plane_stress_stiffness(ElementType.Q4R, 0.0)) == 5
    assert count_zero_energy_modes(single_element_plane_stress_stiffness(ElementType.Q4R, 0.1)) == 3


def test_Q9R_has_spurious_zero_energy_modes():
    """With 2x2 Gauss points, Q9 has 3 spurious zero energy modes in addition to the 3 rigid body modes."""
    assert count_zero_energy_modes(single_element_plane_stress_stiffness(ElementType.Q9R)) == 6


#====================================================================
# Singular systems: The solver prints a warning, but still gives a
# result, so that students can see what happens
#====================================================================


def solve_rectangle_in_tension(element_type, fix_v=True):
    """u = 0 on the west edge and v = 0 at the south-west corner: Just enough to prevent rigid body
    motion. Without the corner condition (fix_v=False), the body can move freely in the y-direction."""
    config = create_config(100.0, 0.25, 1.0, element_type, ProblemType.PLANE_STRESS)
    mesh = create_structured_quad_mesh(config, 2.0, 1.0, 3, 3)
    add_boundary_condition(config, mesh, "west", Dof.U, 0)
    if fix_v:
        add_boundary_condition(config, mesh, "south_west", Dof.V, 0)
    add_load(config, mesh, "east", LoadType.TRACTION, lambda x, y: (3.0, 0.0))
    solver_data = create_solver_data(config, mesh)
    solve(config, solver_data, mesh)


@pytest.mark.parametrize(
    "solve_problem",
    [
        lambda: solve_rectangle_in_tension(ElementType.Q4, fix_v=False),
        lambda: solve_rectangle_in_tension(ElementType.Q9R),
        lambda: solve_square_plate(ElementType.Q9R, 4),
        lambda: solve_square_plate(ElementType.Q4R, 4),  # Q4R only has hourglass stabilization for plane stress
    ],
    ids=["missing boundary condition", "Q9R plane stress", "Q9R plate", "Q4R plate"])
def test_singular_system_gives_a_warning(solve_problem, capsys):
    solve_problem()
    assert "WARNING: The system matrix is singular" in capsys.readouterr().out


@pytest.mark.parametrize("element_type", ALL_ELEMENT_TYPES, ids=element_name)
def test_well_posed_problems_give_no_warning(element_type, capsys):
    solve_cantilever(element_type, 4, 2)
    if element_type != ElementType.Q9R:  #Singular with these supports (see above)
        solve_rectangle_in_tension(element_type)
    if element_type not in [ElementType.Q4R, ElementType.Q9R]:  #Singular for this plate (see above)
        solve_square_plate(element_type, 4)
    assert "singular" not in capsys.readouterr().out


def test_exactly_singular_system_gives_a_warning_and_nan(capsys):
    """This happens e.g. if an element stiffness matrix is zero, like an unfinished Q4_USER element.
    No solution exists, but the program continues, so that the results can be inspected."""
    A = csc_matrix(np.diag([1.0, 0.0, 2.0]))
    r = solve_linear_system(A, np.ones(3))
    assert np.all(np.isnan(r))
    assert "WARNING: The system matrix is singular" in capsys.readouterr().out


#====================================================================
# Equilibrium and comparison with analytical solutions
#====================================================================


@pytest.mark.parametrize("element_type", ALL_ELEMENT_TYPES, ids=element_name)
def test_reaction_forces_balance_the_external_forces(element_type):
    config, mesh, solver_data = solve_cantilever(element_type, 4, 2)
    R_reaction = solver_data.R_int - solver_data.R_ext
    Rx_reaction, Ry_reaction = unpack_solution(config, R_reaction)
    Rx_ext, Ry_ext = unpack_solution(config, solver_data.R_ext)
    assert np.isclose(np.sum(Rx_reaction) + np.sum(Rx_ext), 0.0, atol=1e-6)
    assert np.isclose(np.sum(Ry_reaction) + np.sum(Ry_ext), 0.0, atol=1e-6)
    assert np.isclose(np.sum(Ry_reaction), 1000.0)  #The reaction must carry the whole tip force


def test_solving_twice_gives_the_same_result():
    config, mesh, solver_data = solve_cantilever(ElementType.Q4, 4, 2)
    r_first, R_ext_first = solver_data.r.copy(), solver_data.R_ext.copy()
    solve(config, solver_data, mesh)
    assert np.allclose(solver_data.R_ext, R_ext_first)
    assert np.allclose(solver_data.r, r_first)


@pytest.mark.parametrize("element_type, nEx, nEy", [(ElementType.Q9, 40, 4), (ElementType.Q16, 20, 4)],
                         ids=["Q9", "Q16"])
def test_cantilever_tip_deflection_matches_timoshenko_beam_theory(element_type, nEx, nEy):
    Lx, Ly, h, E, nu, P = 10.0, 1.0, 1.0, 210e9, 0.3, 1000.0
    config, mesh, solver_data = solve_cantilever(element_type, nEx, nEy, Lx, Ly, h, E, nu, P)
    u, v = unpack_solution(config, solver_data.r)
    v_tip = v[mesh.node_sets["north_east"][0]]

    I = h * Ly**3 / 12
    G = E / (2 * (1 + nu))
    shear_correction_factor = 5 / 6
    v_tip_timoshenko = -(P * Lx**3 / (3 * E * I) + P * Lx / (shear_correction_factor * G * Ly * h))
    assert v_tip == pytest.approx(v_tip_timoshenko, rel=0.005)


@pytest.mark.parametrize("clamped, kirchhoff_coefficient", [(False, 0.00406), (True, 0.00126)],
                         ids=["simply supported", "clamped"])
def test_thin_square_plate_center_deflection_matches_kirchhoff_theory(clamped, kirchhoff_coefficient):
    """Q16 is used since it is practically free of shear locking at this mesh size. (Q9 also converges,
    but needs a finer mesh for the clamped case: 0.956 of the Kirchhoff value with 8x8 elements.)"""
    L, h, E, nu, q = 1.0, 0.01, 200e9, 0.3, 1000.0
    config, mesh, solver_data = solve_square_plate(ElementType.Q16, 8, L, h, E, nu, q, clamped)
    w, thetax, thetay = unpack_solution(config, solver_data.r)
    w_center = w[find_node_closest_to(mesh, L / 2, L / 2)]

    D = E * h**3 / (12 * (1 - nu**2))
    w_center_kirchhoff = -kirchhoff_coefficient * q * L**4 / D  #Negative: positive pressure points downwards
    assert w_center == pytest.approx(w_center_kirchhoff, rel=0.01)


def test_fully_integrated_Q4_plate_locks_in_shear():
    """This is expected behaviour (shear locking), and an important teaching point. The test makes sure
    that nobody "fixes" it by accident."""
    L, h, E, nu, q = 1.0, 0.01, 200e9, 0.3, 1000.0
    config, mesh, solver_data = solve_square_plate(ElementType.Q4, 8, L, h, E, nu, q)
    w, thetax, thetay = unpack_solution(config, solver_data.r)
    w_center = w[find_node_closest_to(mesh, L / 2, L / 2)]
    D = E * h**3 / (12 * (1 - nu**2))
    w_center_kirchhoff = -0.00406 * q * L**4 / D
    assert abs(w_center / w_center_kirchhoff) < 0.1


#====================================================================
# Regression tests: The numbers below were computed with the code as
# it was before the clean-up in November 2026. They make sure that
# restructuring the code doesn't change any results. If you change
# the formulation on purpose, the numbers must be updated.
#====================================================================

CANTILEVER_TIP_DEFLECTION_10x2 = {
    ElementType.Q4: -1.354490448807078e-05,
    ElementType.Q4R: -2.483368223649329e-05,
    ElementType.Q8: -1.909204564461871e-05,
    ElementType.Q8R: -1.9117520431254024e-05,
    ElementType.Q9: -1.9117848142803338e-05,
    ElementType.Q9R: -1.9159532179316214e-05,
    ElementType.Q16: -1.914770054896495e-05,
}

# Q4R and Q9R are left out: The simply supported plate is singular with these elements
# (see test_singular_system_gives_a_warning), so their results depend on round-off errors.
PLATE_CENTER_DEFLECTION_4x4 = {
    ElementType.Q4: -2.4374421611534405e-06,
    ElementType.Q8: -0.00020393378935936916,
    ElementType.Q8R: -0.0002262780345961578,
    ElementType.Q9: -0.00021774214739001354,
    ElementType.Q16: -0.00022243574839503469,
}


@pytest.mark.parametrize("element_type", ALL_ELEMENT_TYPES, ids=element_name)
def test_cantilever_result_is_unchanged(element_type):
    config, mesh, solver_data = solve_cantilever(element_type, 10, 2)
    u, v = unpack_solution(config, solver_data.r)
    v_tip = v[mesh.node_sets["north_east"][0]]
    assert v_tip == pytest.approx(CANTILEVER_TIP_DEFLECTION_10x2[element_type], rel=1e-9)


@pytest.mark.parametrize("element_type", PLATE_CENTER_DEFLECTION_4x4.keys(), ids=element_name)
def test_plate_result_is_unchanged(element_type):
    config, mesh, solver_data = solve_square_plate(element_type, 4)
    w, thetax, thetay = unpack_solution(config, solver_data.r)
    w_center = w[find_node_closest_to(mesh, 0.5, 0.5)]
    assert w_center == pytest.approx(PLATE_CENTER_DEFLECTION_4x4[element_type], rel=1e-9)


def test_small_structure_in_metres():
    """A 1 mm x 0.5 mm plate in uniaxial tension, with the lengths in metres. The elements are so small that
    det(J) is about 1e-9, which must not be mistaken for a distorted element. Exact solution: u = sigma0 * x / E."""
    E, sigma0, Lx, Ly = 210e9, 1e6, 1e-3, 0.5e-3
    config = create_config(E, 0.3, 1e-3, ElementType.Q4, ProblemType.PLANE_STRESS)
    mesh = create_structured_quad_mesh(config, Lx, Ly, 20, 10)
    add_boundary_condition(config, mesh, "west", Dof.U, 0)
    add_boundary_condition(config, mesh, "south_west", Dof.V, 0)
    add_load(config, mesh, "east", LoadType.TRACTION, lambda x, y: (sigma0, 0.0))
    solver_data = create_solver_data(config, mesh)
    solve(config, solver_data, mesh)
    u, v = unpack_solution(config, solver_data.r)
    assert np.allclose(u, sigma0 * mesh.nodes[:, 0] / E, rtol=1e-8, atol=1e-18)
