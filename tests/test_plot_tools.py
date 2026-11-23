"""Smoke tests of the GUI and plot_field: Draw every combination of options and check that nothing crashes.
The figures are drawn into memory (see conftest.py), so no windows open."""
import os

import matplotlib.pyplot as plt
import pytest

import fem.plot_tools as plot_tools
import fem.stress_recovery as stress_recovery
from fem.useful_imports import *


@pytest.fixture(autouse=True)
def gui_does_not_block(monkeypatch):
    """Plot() calls plt.show() at the end. Replace it with a function that does nothing,
    and close the figures after each test."""
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    yield
    plt.close("all")


def solved_plane_stress_problem():
    config = create_config(210e9, 0.3, 1.0, ElementType.Q9, ProblemType.PLANE_STRESS)
    mesh = create_structured_quad_mesh(config, 4.0, 1.0, 4, 2)
    add_boundary_condition(config, mesh, "west", Dof.U, 0)
    add_boundary_condition(config, mesh, "west", Dof.V, 0)
    add_load(config, mesh, "north", LoadType.PRESSURE, lambda x, y: 1000.0)
    solver_data = create_solver_data(config, mesh)
    solve(config, solver_data, mesh)
    return config, mesh, solver_data


def solved_plate_problem():
    config = create_config(210e9, 0.3, 0.01, ElementType.Q9, ProblemType.PLATE)
    mesh = create_structured_quad_mesh(config, 1.0, 1.0, 3, 3)
    add_boundary_condition(config, mesh, "west", Dof.W, 0)
    add_boundary_condition(config, mesh, "west", Dof.THETAY, 0)
    add_boundary_condition(config, mesh, "south", Dof.W, 0)
    add_boundary_condition(config, mesh, "south", Dof.THETAX, 0)
    add_load(config, mesh, "domain", LoadType.PRESSURE, lambda x, y: 1000.0)
    solver_data = create_solver_data(config, mesh)
    solve(config, solver_data, mesh)
    return config, mesh, solver_data


def all_contours(problem_type):
    """All fields, and both stress recovery methods for the stresses"""
    return [(field, recovery) for field in plot_tools.get_field_names(problem_type)
            for recovery in (["consistent", "L2"] if plot_tools.is_stress_field(field) else ["consistent"])]


@pytest.mark.parametrize("contour, stress_recovery_method", all_contours(ProblemType.PLANE_STRESS))
def test_plane_stress_contours(contour, stress_recovery_method):
    config, mesh, solver_data = solved_plane_stress_problem()
    config.contour = contour
    config.stress_recovery = stress_recovery_method
    plot_tools.Plot(config, mesh, solver_data)


@pytest.mark.parametrize("contour, stress_recovery_method", all_contours(ProblemType.PLATE))
def test_plate_contours_and_boundary_conditions(contour, stress_recovery_method):
    config, mesh, solver_data = solved_plate_problem()
    config.contour = contour
    config.stress_recovery = stress_recovery_method
    config.show_bcs = True  #The plate problem has w, thetax and thetay boundary conditions
    config.show_node_labels = True
    plot_tools.Plot(config, mesh, solver_data)


@pytest.mark.parametrize("arrow_type", ["none", "disp", "external forces", "reaction forces"])
def test_plane_stress_arrows_labels_and_boundary_conditions(arrow_type):
    config, mesh, solver_data = solved_plane_stress_problem()
    config.arrow_type = arrow_type
    config.show_bcs = True
    config.show_node_labels = True
    config.specify_contour_limits = True
    plot_tools.Plot(config, mesh, solver_data)


def test_unknown_contour_gives_a_clear_error():
    config, mesh, solver_data = solved_plate_problem()
    config.contour = "sigma_xx"  #Not available for plates
    with pytest.raises(ValueError, match="M_xx"):
        plot_tools.Plot(config, mesh, solver_data)


@pytest.mark.parametrize("contour", ["u_mag", "sigma_xx"])
def test_gui_shows_nan_results_without_crashing(contour):
    """An exactly singular system gives NaN results (see test_solver.py). The GUI should still open,
    so that the user can see that something is wrong."""
    config, mesh, solver_data = solved_plane_stress_problem()
    solver_data.r[:] = np.nan
    solver_data.R_int[:] = np.nan
    config.contour = contour
    config.arrow_type = "reaction forces"
    plot_tools.Plot(config, mesh, solver_data)


def test_arrows_with_nothing_to_draw():
    """No external forces, since the structure is only loaded by a prescribed displacement."""
    config = create_config(210e9, 0.3, 1.0, ElementType.Q4, ProblemType.PLANE_STRESS)
    mesh = create_structured_quad_mesh(config, 4.0, 1.0, 4, 2)
    add_boundary_condition(config, mesh, "west", Dof.U, 0)
    add_boundary_condition(config, mesh, "west", Dof.V, 0)
    add_boundary_condition(config, mesh, "east", Dof.U, 0.01)
    solver_data = create_solver_data(config, mesh)
    solve(config, solver_data, mesh)
    config.arrow_type = "external forces"
    plot_tools.Plot(config, mesh, solver_data)


def test_gui_draws_the_plot_only_once(monkeypatch):
    calls = []
    original_plot_field = plot_tools.plot_field

    def counting_plot_field(*args, **kwargs):
        calls.append(1)
        return original_plot_field(*args, **kwargs)

    monkeypatch.setattr(plot_tools, "plot_field", counting_plot_field)
    config, mesh, solver_data = solved_plane_stress_problem()
    plot_tools.Plot(config, mesh, solver_data)
    assert len(calls) == 1


def test_gui_calculates_the_L2_stresses_only_once(monkeypatch):
    calls = []
    original_calc_nodal_stress_L2 = stress_recovery.calc_nodal_stress_L2

    def counting_calc_nodal_stress_L2(*args, **kwargs):
        calls.append(1)
        return original_calc_nodal_stress_L2(*args, **kwargs)

    monkeypatch.setattr(stress_recovery, "calc_nodal_stress_L2", counting_calc_nodal_stress_L2)
    config, mesh, solver_data = solved_plane_stress_problem()
    config.contour = "sigma_xx"
    config.stress_recovery = "L2"
    plot = plot_tools.Plot(config, mesh, solver_data)
    plot.radio_contour.set_active(plot.field_names.index("sigma_vm"))  #Clicking another stress redraws the plot
    assert len(calls) == 1


def test_stress_recovery_is_greyed_out_for_displacements():
    config, mesh, solver_data = solved_plane_stress_problem()
    config.contour = "u"
    plot = plot_tools.Plot(config, mesh, solver_data)
    assert plot.radio_recovery.labels[0].get_color() == "lightgrey"
    plot.radio_contour.set_active(plot.field_names.index("sigma_xx"))  #Clicking a stress redraws the plot
    assert plot.radio_recovery.labels[0].get_color() == "black"


def test_fonts_scale_with_the_window_height():
    config, mesh, solver_data = solved_plane_stress_problem()
    plot = plot_tools.Plot(config, mesh, solver_data)
    plot.fig.set_size_inches(10, 2 * plot.DESIGN_HEIGHT)
    plot.on_resize(None)
    assert all(text.get_fontsize() == pytest.approx(2 * plot.FONT_SIZE) for text in plot.widget_texts)


def test_save_screen_writes_an_image_to_the_current_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config, mesh, solver_data = solved_plane_stress_problem()
    plot = plot_tools.Plot(config, mesh, solver_data)
    plot.save_screen()
    saved_files = os.listdir(tmp_path)
    assert len(saved_files) == 1 and saved_files[0].endswith(".png")


#====================================================================
# plot_field, for use in scripts
#====================================================================


def test_plot_field_in_a_script():
    config, mesh, solver_data = solved_plane_stress_problem()
    config.disp_scaling = 0  #Show the undeformed mesh
    fig, ax = plt.subplots()
    contours = plot_field(ax, config, mesh, solver_data, "v")
    fig.colorbar(contours, ax=ax)
    u, v = unpack_solution(config, solver_data.r)
    #The colors go from the smallest to the largest displacement, which are found at the nodes
    assert contours.get_clim() == pytest.approx((np.min(v), np.max(v)))


def test_plot_field_with_an_unknown_field_gives_a_clear_error():
    config, mesh, solver_data = solved_plane_stress_problem()
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="sigma_xx"):
        plot_field(ax, config, mesh, solver_data, "stress")
