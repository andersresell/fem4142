"""Smoke tests of the GUI and plot_field: Draw every combination of options and check that nothing crashes.
The figures are drawn into memory (see conftest.py), so no windows open."""
import os

import matplotlib.pyplot as plt
import matplotlib.quiver
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
    contours = []
    for field in plot_tools.get_field_names(problem_type):
        contours.append((field, "consistent"))
        if plot_tools.is_stress_field(field):
            contours.append((field, "L2"))
    return contours


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
    config.show_labels = True
    plot_tools.Plot(config, mesh, solver_data)


@pytest.mark.parametrize("arrow_type", ["none", "disp", "external forces", "reaction forces"])
def test_plane_stress_arrows_labels_and_boundary_conditions(arrow_type):
    config, mesh, solver_data = solved_plane_stress_problem()
    config.arrow_type = arrow_type
    config.show_bcs = True
    config.show_labels = True
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


def test_contour_limits_do_not_change_when_they_are_specified():
    """The limits are shown rounded in the text boxes, but ticking "specify contour limits" must keep the exact limits"""
    config, mesh, solver_data = solved_plane_stress_problem()
    config.contour = "sigma_xx"
    plot = plot_tools.Plot(config, mesh, solver_data)
    limits_before = (plot.cbar.norm.vmin, plot.cbar.norm.vmax)
    plot.check_specify_limits.set_active(0)  #Tick "specify contour limits", which redraws the plot
    assert config.specify_contour_limits
    assert (plot.cbar.norm.vmin, plot.cbar.norm.vmax) == limits_before


def test_contour_limit_boxes_are_only_shown_when_the_limits_are_specified():
    config, mesh, solver_data = solved_plane_stress_problem()
    config.contour = "sigma_xx"
    plot = plot_tools.Plot(config, mesh, solver_data)
    assert not plot.box_contour_max.ax.get_visible() and not plot.box_contour_min.ax.get_visible()
    plot.check_specify_limits.set_active(0)  #Tick "specify contour limits"
    assert plot.box_contour_max.ax.get_visible() and plot.box_contour_min.ax.get_visible()


def test_automatic_displacement_scaling_can_be_replaced_by_true_scale():
    config, mesh, solver_data = solved_plane_stress_problem()  #A 4 x 1 mesh
    assert config.disp_scaling == "auto"
    plot = plot_tools.Plot(config, mesh, solver_data)
    u, v = unpack_solution(config, solver_data.r)
    largest_displacement = np.max(np.sqrt(u**2 + v**2))
    assert config.disp_scaling * largest_displacement == pytest.approx(0.1 * 4.0, rel=0.05)  #10% of the mesh size
    assert float(plot.box_disp_scale.text) == config.disp_scaling  #The factor is shown in the text box

    plot.box_disp_scale.set_val("1")  #Typing and pressing enter redraws the plot
    assert config.disp_scaling == 1.0
    plot.box_disp_scale.set_val("auto")
    assert config.disp_scaling * largest_displacement == pytest.approx(0.4, rel=0.05)


def test_disp_scale_box_is_only_shown_for_the_deformed_shape():
    config, mesh, solver_data = solved_plane_stress_problem()
    plot = plot_tools.Plot(config, mesh, solver_data)
    assert config.show_deformed and plot.box_disp_scale.ax.get_visible()
    assert not np.allclose(plot.node_markers.get_offsets(), mesh.nodes)
    plot.check_buttons.set_active(4)  #Untick "deformed shape", which redraws the plot
    assert not config.show_deformed and not plot.box_disp_scale.ax.get_visible()
    assert np.allclose(plot.node_markers.get_offsets(), mesh.nodes)  #The nodes are back at their original positions


def test_plot_field_with_automatic_displacement_scaling():
    config, mesh, solver_data = solved_plane_stress_problem()
    fig, ax = plt.subplots()
    plot_field(ax, config, mesh, solver_data, "u_mag")
    assert config.disp_scaling == "auto"  #plot_field doesn't change config


def solved_clamped_plate_problem():
    config = create_config(210e9, 0.3, 0.01, ElementType.Q9, ProblemType.PLATE)
    mesh = create_structured_quad_mesh(config, 1.0, 1.0, 3, 3)
    for edge in ["west", "east", "south", "north"]:
        for dof in [Dof.W, Dof.THETAX, Dof.THETAY]:
            add_boundary_condition(config, mesh, edge, dof, 0)
    add_load(config, mesh, "domain", LoadType.PRESSURE, lambda x, y: 1000.0)
    solver_data = create_solver_data(config, mesh)
    solve(config, solver_data, mesh)
    return config, mesh, solver_data


CLAMPED = r"$w$, $\theta_x$ and $\theta_y$ prescribed (clamped)"


@pytest.mark.parametrize("solve_problem, expected_labels", [
    (solved_plane_stress_problem, [r"$u$ and $v$ prescribed"]),
    #The corner node between the two edges gets all three conditions, so it is clamped
    (solved_plate_problem, [r"$w$ prescribed", r"$\theta_x$ prescribed", r"$\theta_y$ prescribed", CLAMPED]),
    (solved_clamped_plate_problem, [CLAMPED]),
], ids=["plane stress", "plate", "clamped plate"])
def test_boundary_conditions_are_shown_by_default_and_explained_in_the_legend(solve_problem, expected_labels):
    config, mesh, solver_data = solve_problem()
    plot = plot_tools.Plot(config, mesh, solver_data)
    legend_labels = [text.get_text() for text in plot.ax.get_legend().get_texts()]
    assert sorted(legend_labels) == sorted(expected_labels)


def test_disp_scale_and_arrows_are_only_shown_for_plane_stress():
    config, mesh, solver_data = solved_plate_problem()
    plot = plot_tools.Plot(config, mesh, solver_data)
    assert plot.box_disp_scale is None and plot.radio_arrows is None and plot.slider_arrow_scale is None
    config, mesh, solver_data = solved_plane_stress_problem()
    plot = plot_tools.Plot(config, mesh, solver_data)
    assert plot.box_disp_scale is not None and plot.radio_arrows is not None


def test_arrows_for_a_plate_give_a_clear_error():
    config, mesh, solver_data = solved_plate_problem()
    config.arrow_type = "reaction forces"
    with pytest.raises(ValueError, match="plane stress"):
        plot_tools.Plot(config, mesh, solver_data)


@pytest.mark.parametrize("nE", [1, 4, 30])
def test_the_longest_arrow_follows_the_symbol_size(nE):
    config = create_config(210e9, 0.3, 1.0, ElementType.Q4, ProblemType.PLANE_STRESS)
    mesh = create_structured_quad_mesh(config, 1.0, 1.0, nE, nE)
    add_boundary_condition(config, mesh, "west", Dof.U, 0)
    add_boundary_condition(config, mesh, "west", Dof.V, 0)
    add_load(config, mesh, "east", LoadType.TRACTION, lambda x, y: (1000.0, 0.0))
    solver_data = create_solver_data(config, mesh)
    solve(config, solver_data, mesh)
    config.arrow_type = "external forces"
    plot = plot_tools.Plot(config, mesh, solver_data)
    arrows = [collection for collection in plot.ax.collections if isinstance(collection, matplotlib.quiver.Quiver)][0]
    longest_arrow = np.max(np.sqrt(arrows.U**2 + arrows.V**2)) * plot_tools.get_screen_length_of_unit_length(plot.ax)
    window_scale = plot.fig.get_size_inches()[1] / plot.DESIGN_HEIGHT
    expected_length = max(5 * plot.get_symbol_size(), 25 * window_scale)
    #The arrows make the plot a bit larger after they are drawn, which makes them a bit shorter on the screen
    assert longest_arrow == pytest.approx(expected_length, rel=0.2)
    assert "largest" in plot.ax.get_legend().get_texts()[0].get_text()


def test_nodes_and_element_edges_can_be_hidden_separately():
    config, mesh, solver_data = solved_plane_stress_problem()
    config.show_nodes = False
    plot = plot_tools.Plot(config, mesh, solver_data)
    assert plot.node_markers is None
    plot.check_buttons.set_active(1)  #Tick "nodes"
    plot.check_buttons.set_active(2)  #Untick "element edges"
    assert plot.node_markers is not None
    assert not config.show_element_edges


def test_labels_show_node_and_element_numbers():
    config, mesh, solver_data = solved_plane_stress_problem()
    config.show_labels = True
    plot = plot_tools.Plot(config, mesh, solver_data)
    assert [label.get_text() for label in plot.node_labels] == [str(i) for i in range(mesh.get_nN())]
    assert [label.get_text() for label in plot.element_labels] == [f"({e})" for e in range(mesh.get_nE())]


def test_node_spacing():
    config, mesh, solver_data = solved_plane_stress_problem()  #Q9 elements of size 1 x 0.5
    assert plot_tools.get_node_spacing(mesh, config.element_type) == pytest.approx(0.25)


def test_element_size_uses_the_shortest_of_all_four_sides():
    config = create_config(1.0, 0.3, 1.0, ElementType.Q4, ProblemType.PLANE_STRESS)
    mesh = create_structured_quad_mesh(config, 1.0, 1.0, 1, 1)
    morph_mesh_to_quadrilateral(mesh, [(0, 0), (4, 0), (3, 1), (0, 3)])  #The shortest side goes from (4, 0) to (3, 1)
    assert plot_tools.get_element_size(mesh) == pytest.approx(np.sqrt(2))


def get_element_edge_width(plot):
    element_edges = [collection for collection in plot.ax.collections if collection.get_gid() == "element_edges"]
    return element_edges[0].get_linewidth()[0]


def test_fine_meshes_get_thin_element_edges_and_small_nodes():
    """Otherwise the edges and nodes of a fine mesh would cover the contours"""
    config, mesh, solver_data = solved_plane_stress_problem()  #A coarse mesh
    plot = plot_tools.Plot(config, mesh, solver_data)
    assert get_element_edge_width(plot) == 1.0

    config = create_config(210e9, 0.3, 1.0, ElementType.Q4, ProblemType.PLANE_STRESS)
    mesh = create_structured_quad_mesh(config, 10.0, 1.0, 100, 15)
    add_boundary_condition(config, mesh, "west", Dof.U, 0)
    add_boundary_condition(config, mesh, "west", Dof.V, 0)
    add_load(config, mesh, "north", LoadType.PRESSURE, lambda x, y: 1000.0)
    solver_data = create_solver_data(config, mesh)
    solve(config, solver_data, mesh)
    plot = plot_tools.Plot(config, mesh, solver_data)
    assert get_element_edge_width(plot) < 1.0
    node_diameter = np.sqrt(plot.node_markers.get_sizes()[0])
    assert node_diameter < 0.5 * plot.get_symbol_size()  #Smaller than the nodes of a coarse mesh can be


def test_symbols_have_a_largest_size_and_follow_the_symbol_size_slider():
    """With a single element, the nodes are far apart on the screen, so the symbols would be huge without the limit"""
    config = create_config(1.0, 0.3, 1.0, ElementType.Q4, ProblemType.PLANE_STRESS)
    mesh = create_structured_quad_mesh(config, 1.0, 1.0, 1, 1)
    add_boundary_condition(config, mesh, "west", Dof.U, 0)
    add_boundary_condition(config, mesh, "south_west", Dof.V, 0)
    add_load(config, mesh, "east", LoadType.TRACTION, lambda x, y: (1.0, 0.0))
    solver_data = create_solver_data(config, mesh)
    solve(config, solver_data, mesh)
    config.show_labels = True
    config.show_bcs = True
    plot = plot_tools.Plot(config, mesh, solver_data)
    window_scale = plot.fig.get_size_inches()[1] / plot.DESIGN_HEIGHT
    largest_size = 12 * window_scale
    assert plot.node_labels[0].get_fontsize() == pytest.approx(largest_size)
    assert plot.node_markers.get_sizes()[0] == pytest.approx((0.5 * largest_size)**2)
    assert len(plot.bc_markers) == 2

    plot.slider_symbol_scale.set_val(2.0)  #Moving the slider redraws the plot
    assert plot.node_labels[0].get_fontsize() == pytest.approx(2 * largest_size)


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
    config.show_deformed = False  #Show the undeformed mesh
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


def test_stress_in_an_invalid_element_is_left_out_of_the_plot():
    """A non-convex element has det(J) <= 0 near one corner, where the stress can't be calculated. Those points are
    left empty (NaN), instead of stopping the program."""
    config = create_config(1.0, 0.3, 1.0, ElementType.Q4, ProblemType.PLANE_STRESS)
    mesh = create_structured_quad_mesh(config, 1.0, 1.0, 1, 1)
    mesh.nodes[mesh.node_sets["north_east"][0]] = [0.3, 0.3]  #Pull the corner in, so that the element is not convex
    solver_data = create_solver_data(config, mesh)
    solver_data.r[:] = np.random.default_rng(0).uniform(-1, 1, len(solver_data.r))
    fig, ax = plt.subplots()
    contours = plot_field(ax, config, mesh, solver_data, "sigma_xx")
    values = contours.get_array()
    assert np.any(values.mask) and not np.all(values.mask)  #Some points are left out, but not all


def test_plate_applied_load_contour():
    """The pressure q = 1000 * (1 + x), plus a body force b = 10000 that counts as b * h = 100"""
    h = 0.01
    config = create_config(210e9, 0.3, h, ElementType.Q9, ProblemType.PLATE)
    mesh = create_structured_quad_mesh(config, 1.0, 1.0, 3, 3)
    for edge in ["west", "east", "south", "north"]:
        add_boundary_condition(config, mesh, edge, Dof.W, 0)
    add_load(config, mesh, "domain", LoadType.PRESSURE, lambda x, y: 1000.0 * (1 + x))
    add_load(config, mesh, "domain", LoadType.BODY_FORCE, lambda x, y: 10000.0)
    solver_data = create_solver_data(config, mesh)
    solve(config, solver_data, mesh)
    fig, ax = plt.subplots()
    contours = plot_field(ax, config, mesh, solver_data, "applied_load")
    assert contours.get_clim() == pytest.approx((1000.0 + 100.0, 2000.0 + 100.0))
    assert not plot_tools.is_stress_field("applied_load")


def test_outward_normals():
    config = create_config(1.0, 0.3, 1.0, ElementType.Q9, ProblemType.PLANE_STRESS)
    mesh = create_structured_quad_mesh(config, 2.0, 1.0, 2, 2)
    normals = plot_tools.calc_outward_normals(mesh, config.element_type)
    x, y = mesh.nodes[:, 0], mesh.nodes[:, 1]
    assert normals[mesh.node_sets["north_east"][0]] == pytest.approx([1 / np.sqrt(2), 1 / np.sqrt(2)])
    middle_of_south_edge = np.where(np.isclose(y, 0.0) & np.isclose(x, 1.0))[0][0]
    assert normals[middle_of_south_edge] == pytest.approx([0.0, -1.0])
    inside = (x > 0) & (x < 2) & (y > 0) & (y < 1)
    assert np.all(np.isnan(normals[inside]))


@pytest.mark.parametrize("traction, arrows_start_at_the_edge", [((-1000.0, 0.0), False), ((1000.0, 0.0), True)],
                         ids=["pushing", "pulling"])
def test_forces_that_point_into_the_body_are_drawn_outside_it(traction, arrows_start_at_the_edge):
    """A traction on the east edge (x = 4). Pushing arrows end at the edge, and pulling arrows start at it."""
    config = create_config(210e9, 0.3, 1.0, ElementType.Q9, ProblemType.PLANE_STRESS)
    mesh = create_structured_quad_mesh(config, 4.0, 1.0, 4, 2)
    add_boundary_condition(config, mesh, "west", Dof.U, 0)
    add_boundary_condition(config, mesh, "west", Dof.V, 0)
    add_load(config, mesh, "east", LoadType.TRACTION, lambda x, y: traction)
    solver_data = create_solver_data(config, mesh)
    solve(config, solver_data, mesh)
    config.arrow_type = "external forces"
    config.show_deformed = False  #Keep the nodes on the edge
    plot = plot_tools.Plot(config, mesh, solver_data)
    arrows = [collection for collection in plot.ax.collections if isinstance(collection, matplotlib.quiver.Quiver)][0]
    if arrows_start_at_the_edge:
        assert np.allclose(arrows.X, 4.0)
    else:
        assert np.all(arrows.X > 4.0)
