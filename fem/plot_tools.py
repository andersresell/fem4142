import os
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.collections import LineCollection
from matplotlib.path import Path
from matplotlib.text import Text
from matplotlib.transforms import Bbox
from matplotlib.widgets import Slider, RadioButtons, CheckButtons, TextBox, Button
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
from fem.fem_utils import *
import fem.stress_recovery as stress_recovery
from fem.mesh import Mesh
from fem.utils import Config
from fem.solver_data import *
import fem.shape_functions as shape_functions

#The fields that can be shown as contours, with the labels that are used in the GUI and on the colorbar
FIELD_LABELS = {
    "none": "none",
    "u": r"$u$",
    "v": r"$v$",
    "u_mag": r"$|\mathbf{u}| = \sqrt{u^2 + v^2}$",
    "sigma_xx": r"$\sigma_{xx}$",
    "sigma_yy": r"$\sigma_{yy}$",
    "sigma_xy": r"$\sigma_{xy}$",
    "sigma_vm": r"$\sigma_{\mathrm{vM}}$ (von Mises)",
    "w": r"$w$",
    "theta_x": r"$\theta_x$",
    "theta_y": r"$\theta_y$",
    "M_xx": r"$M_{xx}$",
    "M_yy": r"$M_{yy}$",
    "M_xy": r"$M_{xy}$",
    "V_x": r"$V_x$",
    "V_y": r"$V_y$",
}

#The column in the solution (one row per node) that each displacement field is taken from
DOF_FIELDS = {"u": 0, "v": 1, "w": 0, "theta_x": 1, "theta_y": 2}

COLORMAP = plt.get_cmap("turbo").with_extremes(over="lightgrey", under="lightgrey")
GRAY_COLORMAP = mcolors.ListedColormap(["grey"])  #Used when no field is shown

#Symbols for the boundary conditions, with the node at (0, 0). Dof.V: A triangle with its tip at the node and a line
#underneath. Dof.U: The same, rotated 90 degrees. Dof.THETAX and Dof.THETAY: A vertical and a horizontal line.
M, L = Path.MOVETO, Path.LINETO  #Move the pen, or draw a line
BC_SYMBOLS = {
    Dof.V: Path([(0, 0), (-0.6, -0.8), (0.6, -0.8), (0, 0), (-0.6, -1), (0.6, -1)], [M, L, L, L, M, L]),
    Dof.U: Path([(0, 0), (-0.8, 0.6), (-0.8, -0.6), (0, 0), (-1, 0.6), (-1, -0.6)], [M, L, L, L, M, L]),
    Dof.THETAX: Path([(0, -1), (0, 1)], [M, L]),
    Dof.THETAY: Path([(-1, 0), (1, 0)], [M, L]),
}


def get_field_names(problem_type):
    """The fields that can be plotted for each problem type"""
    if problem_type == ProblemType.PLANE_STRESS:
        return ["none", "u", "v", "u_mag", "sigma_xx", "sigma_yy", "sigma_xy", "sigma_vm"]
    else:
        assert problem_type == ProblemType.PLATE
        return ["none", "w", "theta_x", "theta_y", "M_xx", "M_yy", "M_xy", "V_x", "V_y"]


def is_stress_field(field):
    return field not in DOF_FIELDS and field not in ["none", "u_mag"]


#====================================================================
# Contour plots of fields
#====================================================================


def get_num_subdivisions(nE):
    """Each element is drawn as an n x n grid of small quads, so that curved elements and fields look smooth.
    We use many small quads when there are few elements, and fewer when there are many, so that the plot stays fast."""
    TARGET_NUM_POINTS = 20000  #Plot points in total
    return int(np.clip(np.sqrt(TARGET_NUM_POINTS / nE) - 1, 2, 30))


def create_plot_grid(n):
    """Creates the n x n grid of small quads that each element is drawn with (each quad is two triangles).
    Returns the natural coordinates of the grid points, the triangles and the outline of the element,
    given by the local numbers of the grid points, which are the same for all elements."""
    xi_eta = [(-1 + 2 * i / n, -1 + 2 * j / n) for i in range(n + 1) for j in range(n + 1)]

    def point(i, j):
        return i * (n + 1) + j

    triangles = []
    for i in range(n):
        for j in range(n):
            triangles.append([point(i, j), point(i + 1, j), point(i + 1, j + 1)])
            triangles.append([point(i, j), point(i + 1, j + 1), point(i, j + 1)])
    #The outline goes counter-clockwise around the element, and ends where it started
    outline = ([point(i, 0) for i in range(n)] + [point(n, j) for j in range(n)] +
               [point(n - i, n) for i in range(n)] + [point(0, n - j) for j in range(n + 1)])
    return xi_eta, np.array(triangles), np.array(outline)


def calc_field_in_element(config: Config, mesh: Mesh, solver_data: SolverData, field, e, N, dNdxi, dNdeta,
                          nodal_stress_L2):
    """Calculates the field in element e at the points where the shape functions N, dNdxi, dNdeta are given"""
    NUM_DOFS = get_num_dofs_from_problem_type(config.problem_type)
    nodeIDs = mesh.elements[e, :]
    r_e = solver_data.r.reshape(-1, NUM_DOFS)[nodeIDs]  #One row per element node: (u, v) or (w, thetax, thetay)

    if field == "none":
        return np.zeros(len(N))
    if field in DOF_FIELDS:
        return N @ r_e[:, DOF_FIELDS[field]]
    if field == "u_mag":
        return np.sqrt((N @ r_e[:, 0])**2 + (N @ r_e[:, 1])**2)

    #The rest of the fields are stresses
    if config.stress_recovery == "consistent":
        stress = stress_recovery.calc_element_stress(config, mesh, solver_data.r, e, N, dNdxi, dNdeta)
    else:
        assert config.stress_recovery == "L2"
        stress = N @ nodal_stress_L2[nodeIDs]  #Interpolate the nodal stresses with the shape functions
    if field == "sigma_vm":
        return stress_recovery.calc_von_mises(stress)
    return stress[:, stress_recovery.get_stress_names(config.problem_type).index(field)]


def plot_field(ax, config: Config, mesh: Mesh, solver_data: SolverData, field, nodal_stress_L2=None):
    """Draws a field (e.g. "u_mag" or "sigma_xx", see get_field_names) as a contour plot on the mesh, in the
    matplotlib axes ax. For plane stress, the mesh is deformed by the displacements times config.disp_scaling.
    Stresses are calculated with config.stress_recovery ("consistent" or "L2"). The element edges are drawn if
    config.show_element_edges is True. Returns the contour plot, which can be used to add a colorbar. Example:

        fig, ax = plt.subplots()
        contours = plot_field(ax, config, mesh, solver_data, "sigma_xx")
        fig.colorbar(contours, ax=ax)
        plt.show()

    (If the L2 stresses are already calculated with stress_recovery.calc_nodal_stress_L2, they can be given as
    nodal_stress_L2, so that they don't need to be calculated again.)"""
    field_names = get_field_names(config.problem_type)
    if field not in field_names:
        raise ValueError(f"Unknown field '{field}' for {config.problem_type.name}. Available: {', '.join(field_names)}")
    if config.stress_recovery not in ["consistent", "L2"]:
        raise ValueError(f"config.stress_recovery must be 'consistent' or 'L2', got '{config.stress_recovery}'")
    if is_stress_field(field) and config.stress_recovery == "L2" and nodal_stress_L2 is None:
        nodal_stress_L2 = stress_recovery.calc_nodal_stress_L2(config, mesh, solver_data.r)

    #The shape functions at the plot points are the same for all elements, so we calculate them only once
    xi_eta, triangles, outline = create_plot_grid(get_num_subdivisions(mesh.get_nE()))
    N, dNdxi, dNdeta = stress_recovery.calc_N_at_points(xi_eta, config.element_type)

    #Calculate the position of the plot points and the field values there, element by element
    nE = mesh.get_nE()
    x = np.zeros((nE, len(xi_eta)))
    y = np.zeros_like(x)
    values = np.zeros_like(x)
    NUM_DOFS = get_num_dofs_from_problem_type(config.problem_type)
    r = solver_data.r.reshape(-1, NUM_DOFS)  #One row per node
    for e in range(nE):
        nodeIDs = mesh.elements[e, :]
        x[e] = N @ mesh.nodes[nodeIDs, 0]
        y[e] = N @ mesh.nodes[nodeIDs, 1]
        if config.problem_type == ProblemType.PLANE_STRESS:  #Show the deformed shape
            x[e] += config.disp_scaling * (N @ r[nodeIDs, 0])
            y[e] += config.disp_scaling * (N @ r[nodeIDs, 1])
        values[e] = calc_field_in_element(config, mesh, solver_data, field, e, N, dNdxi, dNdeta, nodal_stress_L2)

    #The plot points of element e are numbered from e * (number of plot points per element)
    first_point_of_element = len(xi_eta) * np.arange(nE)
    all_triangles = (triangles[None, :, :] + first_point_of_element[:, None, None]).reshape(-1, 3)
    triangulation = tri.Triangulation(x.ravel(), y.ravel(), all_triangles)
    if field == "none":
        contours = ax.tripcolor(triangulation, values.ravel(), shading="gouraud", cmap=GRAY_COLORMAP)
    else:
        contours = ax.tripcolor(triangulation, values.ravel(), shading="gouraud", cmap=COLORMAP)

    if config.show_element_edges:
        element_outlines = [np.column_stack([x[e, outline], y[e, outline]]) for e in range(nE)]
        ax.add_collection(LineCollection(element_outlines, colors="black", linewidths=1.0))
    ax.set_aspect("equal")
    return contours


#====================================================================
# The GUI
#====================================================================


def get_node_spacing(mesh: Mesh, element_type):
    """A typical distance between neighbouring nodes, used to choose the size of the nodes and labels"""
    corners = mesh.nodes[mesh.elements[:, :4]]  #The four corners of each element
    edge_a = np.linalg.norm(corners[:, 1] - corners[:, 0], axis=1)
    edge_b = np.linalg.norm(corners[:, 3] - corners[:, 0], axis=1)
    shortest_edge = np.median(np.minimum(edge_a, edge_b))
    return shortest_edge / (ELEMENT_TRAITS[element_type].nNl_1D - 1)


def format_contour_limit(value):
    return f"{value:.3e}"


def maximize_window(fig):
    """Makes the window fill the screen. How this is done depends on the GUI toolkit that matplotlib uses,
    so we try the common ones, and keep the window as it is if none of them works."""
    window = getattr(fig.canvas.manager, "window", None)
    ways_to_maximize = [
        lambda: window.state("zoomed"),  #Tk on Windows and macOS
        lambda: window.attributes("-zoomed", True),  #Tk on Linux
        lambda: window.showMaximized(),  #Qt
        lambda: window.maximize(),  #GTK
    ]
    for maximize in ways_to_maximize:
        try:
            maximize()
            return
        except Exception:
            pass


class Plot:
    """The GUI, which shows the mesh and the results, with buttons for choosing what to show.
    Simply construct an instance of the class to start the GUI: Plot(config, mesh, solver_data)"""

    #Font sizes for a window that is DESIGN_HEIGHT inches tall. They are scaled with the actual height of
    #the window, so that the GUI looks the same on small and large screens.
    DESIGN_HEIGHT = 10
    FONT_SIZE = 11  #Buttons and other widgets
    LABEL_FONTSIZE = 20  #Axis labels and colorbar label
    TICK_FONTSIZE = 16  #Numbers on the axes and the colorbar

    def __init__(self, config: Config, mesh: Mesh, solver_data: SolverData):
        self.config = config
        self.mesh = mesh
        self.solver_data = solver_data
        self.nodal_stress_L2 = None  #Calculated the first time an L2 stress is shown
        self.cbar = None
        self.node_markers = None  #The nodes, labels and boundary condition symbols are created in update()
        self.node_labels = []
        self.element_labels = []
        self.bc_markers = []
        self.node_spacing = get_node_spacing(mesh, config.element_type)

        self.field_names = get_field_names(config.problem_type)
        if config.contour not in self.field_names:
            raise ValueError(f"Unknown config.contour '{config.contour}' for {config.problem_type.name}. "
                             f"Available: {', '.join(self.field_names)}")

        self.fig, self.ax = plt.subplots(figsize=(11, 8))
        plt.subplots_adjust(left=0.3, bottom=0.25)
        self.create_widgets()
        problem_name = "Plane stress" if config.problem_type == ProblemType.PLANE_STRESS else "Mindlin plate"
        self.title = self.fig.suptitle(f"{problem_name} problem: {config.element_type.name} element")

        self.fig.canvas.mpl_connect("resize_event", self.on_resize)
        maximize_window(self.fig)
        self.update()
        plt.show()

    def create_widgets(self):
        config = self.config

        #The widgets on the left side are placed from the top and down
        self.top_of_next_widget = 0.93

        def get_widget_axes(left, width, height):
            ax = plt.axes([left, self.top_of_next_widget - height, width, height])
            self.top_of_next_widget -= height + 0.035
            return ax

        self.contour_labels = [FIELD_LABELS[field] for field in self.field_names]
        self.radio_contour = RadioButtons(get_widget_axes(0.03, 0.2, 0.22), self.contour_labels,
                                          active=self.field_names.index(config.contour))
        self.radio_contour.ax.set_title("Contour:")

        recovery_options = ["consistent", "L2"]
        self.radio_recovery = RadioButtons(get_widget_axes(0.03, 0.2, 0.05), recovery_options,
                                           active=recovery_options.index(config.stress_recovery))
        self.radio_recovery.ax.set_title("Stress recovery:")

        if config.problem_type == ProblemType.PLANE_STRESS:
            arrow_options = ["none", "disp", "external forces", "reaction forces"]
        else:
            arrow_options = ["none"]  #Arrows are not implemented for plates
        self.radio_arrows = RadioButtons(get_widget_axes(0.03, 0.2, 0.02 * len(arrow_options)), arrow_options,
                                         active=arrow_options.index(config.arrow_type))
        self.radio_arrows.ax.set_title("Arrows:")

        self.check_buttons = CheckButtons(
            get_widget_axes(0.03, 0.2, 0.08),
            ["show labels", "show nodes", "show element edges", "show boundary conditions"],
            [config.show_labels, config.show_nodes, config.show_element_edges, config.show_bcs])
        self.check_buttons.ax.set_title("Show:")

        self.box_disp_scale = None
        if config.problem_type == ProblemType.PLANE_STRESS:  #Plates are drawn undeformed
            self.box_disp_scale = TextBox(get_widget_axes(0.13, 0.08, 0.03), "disp scale ",
                                          initial=config.disp_scaling)

        #The contour limit boxes are only shown when "fixed contour limits" is ticked (see update)
        self.check_fixed_limits = CheckButtons(get_widget_axes(0.03, 0.2, 0.03), ["fixed contour limits"],
                                               [config.fixed_contour_limits])
        self.box_contour_max = TextBox(get_widget_axes(0.13, 0.08, 0.03), "max ", initial=config.contour_max)
        self.box_contour_min = TextBox(get_widget_axes(0.13, 0.08, 0.03), "min ", initial=config.contour_min)

        #The widgets at the bottom
        self.slider_symbol_scale = Slider(plt.axes([0.4, 0.13, 0.45, 0.03]), "symbol size", 0, 5,
                                          valinit=config.symbol_scale)
        self.slider_arrow_scale = Slider(plt.axes([0.4, 0.08, 0.45, 0.03]), "arrow size", 0, 10,
                                         valinit=config.arrow_scale)
        self.save_button = Button(plt.axes([0.4, 0.02, 0.15, 0.04]), "Save Screen", color="lightblue",
                                  hovercolor="skyblue")
        self.save_button.on_clicked(self.save_screen)

        for radio in [self.radio_contour, self.radio_recovery, self.radio_arrows]:
            radio.on_clicked(self.update)
        self.check_buttons.on_clicked(self.update)
        self.check_fixed_limits.on_clicked(self.update)
        self.box_contour_max.on_submit(self.update)
        self.box_contour_min.on_submit(self.update)
        if self.box_disp_scale is not None:
            self.box_disp_scale.on_submit(self.update)
        self.slider_symbol_scale.on_changed(self.update)
        self.slider_arrow_scale.on_changed(self.update)

        #All the text in the widgets, so that the font size can be changed when the window changes size
        widgets = [
            self.radio_contour, self.radio_recovery, self.radio_arrows, self.check_buttons, self.box_disp_scale,
            self.check_fixed_limits, self.box_contour_max, self.box_contour_min, self.slider_symbol_scale,
            self.slider_arrow_scale, self.save_button
        ]
        self.widget_texts = [text for widget in widgets if widget is not None for text in widget.ax.findobj(Text)]

    def read_widgets(self):
        """Copies the choices in the GUI to config"""
        config = self.config
        config.contour = self.field_names[self.contour_labels.index(self.radio_contour.value_selected)]
        config.stress_recovery = self.radio_recovery.value_selected
        config.arrow_type = self.radio_arrows.value_selected
        config.show_labels, config.show_nodes, config.show_element_edges, config.show_bcs = \
            self.check_buttons.get_status()
        config.fixed_contour_limits = self.check_fixed_limits.get_status()[0]
        config.symbol_scale = self.slider_symbol_scale.val
        config.arrow_scale = self.slider_arrow_scale.val
        if self.box_disp_scale is not None:
            try:
                config.disp_scaling = float(self.box_disp_scale.text)
            except ValueError:
                print(f"Invalid input for the displacement scaling: '{self.box_disp_scale.text}'. Using 1 instead.")
                config.disp_scaling = 1.0

        #The contour limits are shown rounded in the text boxes. They are only read if the user has changed the
        #text, so that ticking "fixed" doesn't replace the exact limits by the rounded ones.
        for box, name in [(self.box_contour_min, "contour_min"), (self.box_contour_max, "contour_max")]:
            if box.text != format_contour_limit(getattr(config, name)):
                try:
                    setattr(config, name, float(box.text))
                except ValueError:
                    print(f"Invalid input for the contour limit: '{box.text}'. Keeping the previous limit.")

    def update(self, event=None):
        """Reads the choices in the GUI and draws everything again"""
        self.read_widgets()
        config = self.config
        self.ax.cla()

        #====================================================================
        # Contour plot. The L2 stresses are only calculated once.
        #====================================================================
        if is_stress_field(config.contour) and config.stress_recovery == "L2" and self.nodal_stress_L2 is None:
            self.nodal_stress_L2 = stress_recovery.calc_nodal_stress_L2(config, self.mesh, self.solver_data.r)
        contours = plot_field(self.ax, config, self.mesh, self.solver_data, config.contour, self.nodal_stress_L2)
        self.update_contour_limits_and_colorbar(contours)

        #The stress recovery only matters for stresses, so it is greyed out when another field is shown
        recovery_color = "black" if is_stress_field(config.contour) else "lightgrey"
        for text in list(self.radio_recovery.labels) + [self.radio_recovery.ax.title]:
            text.set_color(recovery_color)

        #The boxes for the contour limits are only shown when the limits are fixed (otherwise, see the colorbar)
        self.box_contour_max.ax.set_visible(config.fixed_contour_limits)
        self.box_contour_min.ax.set_visible(config.fixed_contour_limits)

        #====================================================================
        # Nodes, arrows and boundary conditions
        #====================================================================
        if config.problem_type == ProblemType.PLANE_STRESS:
            u, v = unpack_solution(config, self.solver_data.r)
            x_nodes = self.mesh.nodes[:, 0] + config.disp_scaling * u
            y_nodes = self.mesh.nodes[:, 1] + config.disp_scaling * v
        else:
            x_nodes = self.mesh.nodes[:, 0]
            y_nodes = self.mesh.nodes[:, 1]
        domain_scale = 20 / np.sqrt(self.mesh.get_nN())  #Smaller arrows when there are many nodes
        self.plot_nodes_and_labels(x_nodes, y_nodes)
        self.plot_arrow_field(x_nodes, y_nodes, domain_scale)
        self.plot_bcs(x_nodes, y_nodes)

        self.ax.set_xlabel(r"$x$")
        self.ax.set_ylabel(r"$y$")
        self.apply_sizes()
        self.fig.canvas.draw_idle()

    def update_contour_limits_and_colorbar(self, contours):
        config = self.config
        if config.contour == "none":
            if self.cbar is not None:
                self.cbar.ax.set_visible(False)
            return

        if config.fixed_contour_limits:
            contours.set_clim(config.contour_min, config.contour_max)
        else:
            #Remember the limits, and show them in the text boxes. The events must be switched off while doing this,
            #since changing the text otherwise triggers update(), which would draw the whole plot again.
            config.contour_min, config.contour_max = contours.get_clim()
            for text_box, value in [(self.box_contour_min, config.contour_min), (self.box_contour_max, config.contour_max)]:
                text_box.eventson = False
                text_box.set_val(format_contour_limit(value))
                text_box.eventson = True

        if self.cbar is None:
            self.cbar = self.fig.colorbar(contours, ax=self.ax)
        self.cbar.ax.set_visible(True)
        self.cbar.update_normal(contours)
        vmin, vmax = contours.get_clim()
        self.cbar.set_ticks(np.linspace(vmin, vmax, 10))  #Uniformly spaced ticks
        self.cbar.set_label(FIELD_LABELS[config.contour])

    def apply_sizes(self):
        """Sets all font sizes in proportion to the height of the window, and the sizes of the nodes, labels and
        boundary condition symbols from the symbol size"""
        scale = self.fig.get_size_inches()[1] / self.DESIGN_HEIGHT
        for text in self.widget_texts:
            text.set_fontsize(self.FONT_SIZE * scale)
        self.title.set_fontsize(1.2 * self.FONT_SIZE * scale)
        self.ax.xaxis.label.set_fontsize(self.LABEL_FONTSIZE * scale)
        self.ax.yaxis.label.set_fontsize(self.LABEL_FONTSIZE * scale)
        self.ax.tick_params(labelsize=self.TICK_FONTSIZE * scale)
        if self.cbar is not None:
            self.cbar.ax.yaxis.label.set_fontsize(self.LABEL_FONTSIZE * scale)
            self.cbar.ax.tick_params(labelsize=self.TICK_FONTSIZE * scale)
        legend = self.ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(self.FONT_SIZE * scale)

        #Scatter plots take the marker size as an area (in points^2)
        size = self.get_symbol_size()
        if self.node_markers is not None:
            self.node_markers.set_sizes([(0.5 * size)**2])
        for label in self.node_labels:
            label.set_fontsize(size)
            label.xyann = (0.25 * size, 0.25 * size)  #Distance from the node (in points)
        for label in self.element_labels:
            label.set_fontsize(size)
            label.xyann = (0, -0.4 * size)  #Below the center (in points)
        for markers, relative_size in self.bc_markers:
            markers.set_sizes([(relative_size * size)**2])
            markers.set_linewidths(max(0.8, 0.12 * size))

    def on_resize(self, event):
        self.apply_sizes()
        self.fig.canvas.draw_idle()

    def save_screen(self, event=None):
        """Saves the plot area (the axes and the colorbar, without the GUI widgets) as a high resolution
        image in the folder that the script was run from."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = os.path.join(os.getcwd(), f'fem_plot_{timestamp}.png')

        # Find the bounding box of the axes and the colorbar (including labels and title) in pixels
        renderer = self.fig.canvas.get_renderer()
        extent = self.ax.get_tightbbox(renderer)
        if self.cbar is not None and self.cbar.ax.get_visible():
            extent = Bbox.union([extent, self.cbar.ax.get_tightbbox(renderer)])
        extent = extent.transformed(self.fig.dpi_scale_trans.inverted())  #Convert from pixels to inches

        self.fig.savefig(filepath, dpi=600, bbox_inches=extent)
        print(f"Saved the plot to {filepath}")

    #====================================================================
    # Nodes, labels and boundary conditions. They are drawn here, and
    # their sizes are set in apply_sizes(), since the sizes must be
    # updated when the window changes size.
    #====================================================================

    def plot_nodes_and_labels(self, x_nodes, y_nodes):
        config = self.config
        self.node_markers = None
        if config.show_nodes:
            self.node_markers = self.ax.scatter(x_nodes, y_nodes, color="black", zorder=10)

        self.node_labels = []
        self.element_labels = []
        if config.show_labels:
            for i in range(len(x_nodes)):
                #The label is placed up and to the right of the node, at a distance set in apply_sizes
                self.node_labels.append(
                    self.ax.annotate(str(i), (x_nodes[i], y_nodes[i]), xytext=(0, 0), textcoords="offset points",
                                     zorder=20))
            #The element numbers are shown in parentheses just below the center of the elements (xi = eta = 0),
            #so that they don't cover the center node of Q9 elements
            N_center = shape_functions.calc_N(0.0, 0.0, config.element_type)
            for e in range(self.mesh.get_nE()):
                nodeIDs = self.mesh.elements[e, :]
                center = (N_center @ x_nodes[nodeIDs], N_center @ y_nodes[nodeIDs])
                self.element_labels.append(
                    self.ax.annotate(f"({e})", center, xytext=(0, 0), textcoords="offset points", ha="center",
                                     va="top", zorder=20))

    def plot_bcs(self, x_nodes, y_nodes):
        """Draws a symbol at each node with a boundary condition"""
        self.bc_markers = []  #Pairs of (markers, size relative to the symbol size), used in apply_sizes
        if not self.config.show_bcs:
            return
        for bc in self.config.bcs:
            nodeIDs = self.mesh.node_sets[bc.node_set_name]
            if bc.dof == Dof.W:
                markers = self.ax.scatter(x_nodes[nodeIDs], y_nodes[nodeIDs], marker="o", color="magenta", zorder=9)
                self.bc_markers.append((markers, 0.9))
            else:
                markers = self.ax.scatter(x_nodes[nodeIDs], y_nodes[nodeIDs], marker=BC_SYMBOLS[bc.dof],
                                          facecolors="none", edgecolors="magenta", zorder=15)
                self.bc_markers.append((markers, 2.8 if bc.dof in [Dof.U, Dof.V] else 2.0))

    def get_symbol_size(self):
        """The size (in points, 1/72 inch) of the nodes, labels and boundary condition symbols. It follows the distance
        between the nodes on the screen, but is kept between a smallest and a largest size (which follow the window
        size, like the fonts). The symbol size slider scales it."""
        window_scale = self.fig.get_size_inches()[1] / self.DESIGN_HEIGHT
        node_spacing_on_screen = self.node_spacing * self.get_screen_length_of_unit_length()
        return np.clip(0.35 * node_spacing_on_screen, 3 * window_scale, 12 * window_scale) * self.config.symbol_scale

    def get_screen_length_of_unit_length(self):
        """The length on the screen (in points, 1/72 inch) of one unit of length in the plot"""
        self.ax.autoscale_view()
        self.ax.apply_aspect()
        x0 = self.ax.transData.transform((0, 0))[0]
        x1 = self.ax.transData.transform((1, 0))[0]
        return (x1 - x0) * 72 / self.fig.dpi

    def plot_arrow_field(self, x_nodes, y_nodes, domain_scale):
        config = self.config
        if config.arrow_type == "none":
            return
        assert config.problem_type == ProblemType.PLANE_STRESS, "Arrows are not implemented for plate problems"
        if config.arrow_type == "disp":
            u, v = unpack_solution(config, self.solver_data.r)
            self.plot_arrows(x_nodes, y_nodes, u, v, "disp", "blue", domain_scale, config.arrow_scale)
        elif config.arrow_type == "external forces":
            Rx_ext, Ry_ext = unpack_solution(config, self.solver_data.R_ext)
            self.plot_arrows(x_nodes, y_nodes, Rx_ext, Ry_ext, "external forces", "red", domain_scale,
                             config.arrow_scale)
        elif config.arrow_type == "reaction forces":
            Rx_rea, Ry_rea = unpack_solution(config, self.solver_data.R_int - self.solver_data.R_ext)
            self.plot_arrows(x_nodes, y_nodes, Rx_rea, Ry_rea, "reaction forces", "green", domain_scale,
                             config.arrow_scale)
        else:
            raise ValueError(f"Unknown arrow type for plotting: {config.arrow_type}")

    def plot_arrows(self, x, y, field_x, field_y, label, color, domain_scale, user_scale):
        #only plot arrows with a non negligible magnitude
        values_keep = np.where(np.sqrt(field_x**2 + field_y**2) > SMALL_VAL)[0]
        if len(values_keep) == 0:  #Nothing to draw, e.g. no external forces, or NaN from a singular system
            return
        field_x_keep = field_x[values_keep]
        field_y_keep = field_y[values_keep]
        x_keep = x[values_keep]
        y_keep = y[values_keep]

        ARROW_LENGTH_SCALE = 0.05
        ARROW_WIDTH_SCALE = 0.002
        max_vector_norm = np.max(np.sqrt(field_x_keep**2 + field_y_keep**2))
        self.ax.quiver(
            x_keep,
            y_keep,
            ARROW_LENGTH_SCALE * user_scale * field_x_keep / max_vector_norm,
            ARROW_LENGTH_SCALE * user_scale * field_y_keep / max_vector_norm,
            color=color,
            scale=1,  #this scale thing is weird, increasing it makes arrows smaller
            width=ARROW_WIDTH_SCALE * domain_scale)

        # Create a proxy handle for the legend
        arrow_proxy = mlines.Line2D([], [],
                                    color=color,
                                    marker=r'$\rightarrow$',
                                    linestyle='None',
                                    markersize=10,
                                    label=label)
        self.ax.legend(handles=[arrow_proxy])

