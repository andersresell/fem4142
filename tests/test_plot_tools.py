"""Smoke tests of the GUI: Draw every combination of options and check that nothing crashes.
The figures are drawn into memory (see conftest.py), so no windows open."""
import os

import matplotlib.pyplot as plt
import pytest

import fem.plot_tools as plot_tools
from fem.useful_imports import *


@pytest.fixture(autouse=True)
def gui_does_not_block(monkeypatch):
    """Plot() calls plt.show() at the end. Replace it with a function that does nothing,
    and close the figures after each test."""
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    yield
    plt.close("all")


def solved_plane_stress_problem():
    config = create_config(210e9, 0.3, 1.0, ELEMENT_TYPE_Q9, PROBLEM_TYPE_PLANE_STRESS)
    mesh = create_structured_quad_mesh(config, 4.0, 1.0, 4, 2)
    add_boundary_condition(config, mesh, "west", DOF_U, 0)
    add_boundary_condition(config, mesh, "west", DOF_V, 0)
    add_load(config, mesh, "north", LOAD_TYPE_PRESSURE, lambda x, y: 1000.0)
    solver_data = create_solver_data(config, mesh)
    solve(config, solver_data, mesh)
    return config, mesh, solver_data


def solved_plate_problem():
    config = create_config(210e9, 0.3, 0.01, ELEMENT_TYPE_Q9, PROBLEM_TYPE_PLATE)
    mesh = create_structured_quad_mesh(config, 1.0, 1.0, 3, 3)
    add_boundary_condition(config, mesh, "west", DOF_W, 0)
    add_boundary_condition(config, mesh, "west", DOF_THETAY, 0)
    add_boundary_condition(config, mesh, "south", DOF_W, 0)
    add_boundary_condition(config, mesh, "south", DOF_THETAX, 0)
    add_load(config, mesh, "domain", LOAD_TYPE_PRESSURE, lambda x, y: 1000.0)
    solver_data = create_solver_data(config, mesh)
    solve(config, solver_data, mesh)
    return config, mesh, solver_data


PLANE_STRESS_CONTOURS = [("none", None)] + [("disp", c) for c in ["mag", "u", "v"]] + \
                        [("stress", c) for c in ["mises", "xx", "yy", "xy"]]


@pytest.mark.parametrize("contour_type, component", PLANE_STRESS_CONTOURS)
def test_plane_stress_contours(contour_type, component):
    config, mesh, solver_data = solved_plane_stress_problem()
    config.contour_type = contour_type
    if contour_type == "disp":
        config.disp_component = component
    elif contour_type == "stress":
        config.stress_component = component
    plot_tools.Plot(config, mesh, solver_data)


@pytest.mark.parametrize("arrow_type", ["none", "disp", "external forces", "reaction forces"])
def test_plane_stress_arrows_labels_and_boundary_conditions(arrow_type):
    config, mesh, solver_data = solved_plane_stress_problem()
    config.arrow_type = arrow_type
    config.show_bcs = True
    config.show_node_labels = True
    plot_tools.Plot(config, mesh, solver_data)


@pytest.mark.parametrize("component", ["w", "thetax", "thetay"])
def test_plate_contours_and_boundary_conditions(component):
    config, mesh, solver_data = solved_plate_problem()
    config.contour_type = "disp"
    config.disp_component = component
    config.show_bcs = True  #The plate problem has w, thetax and thetay boundary conditions
    config.show_node_labels = True
    plot_tools.Plot(config, mesh, solver_data)


def test_plate_stress_contour_gives_a_clear_error_until_it_is_implemented():
    config, mesh, solver_data = solved_plate_problem()
    config.contour_type = "stress"
    with pytest.raises(ValueError, match="plate"):
        plot_tools.Plot(config, mesh, solver_data)


def test_gui_draws_the_plot_only_once(monkeypatch):
    calls = []
    original_plot_2d = plot_tools.Plot.plot_2d

    def counting_plot_2d(self, *args, **kwargs):
        calls.append(1)
        return original_plot_2d(self, *args, **kwargs)

    monkeypatch.setattr(plot_tools.Plot, "plot_2d", counting_plot_2d)
    config, mesh, solver_data = solved_plane_stress_problem()
    config.contour_type = "stress"
    plot_tools.Plot(config, mesh, solver_data)
    assert len(calls) == 1


def test_save_screen_writes_an_image_to_the_current_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config, mesh, solver_data = solved_plane_stress_problem()
    plot = plot_tools.Plot(config, mesh, solver_data)
    plot.save_screen()
    saved_files = os.listdir(tmp_path)
    assert len(saved_files) == 1 and saved_files[0].endswith(".png")
