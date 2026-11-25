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
import fem.loads as loads
from fem.mesh import Mesh
from fem.utils import Config
from fem.solver_data import *
import fem.shape_functions as shape_functions

#The fields that can be shown as contours, with the labels that are used in the GUI and on the colorbar
FIELD_LABELS = {
    "none": "none",
    "u": r"$u$",
    "v": r"$v$",
    "u_mag": r"$\|\mathbf{u}\|$",
    "sigma_xx": r"$\sigma_{xx}$",
    "sigma_yy": r"$\sigma_{yy}$",
    "sigma_xy": r"$\sigma_{xy}$",
    "sigma_vm": r"$\sigma_{\mathrm{VM}}$",
    "w": r"$w$",
    "theta_x": r"$\theta_x$",
    "theta_y": r"$\theta_y$",
    "M_xx": r"$M_{xx}$",
    "M_yy": r"$M_{yy}$",
    "M_xy": r"$M_{xy}$",
    "V_x": r"$V_x$",
    "V_y": r"$V_y$",
    "applied_load": r"applied load $p$ (force per area)",
}

#The column in the solution (one row per node) that each displacement field is taken from
DOF_FIELDS = {"u": 0, "v": 1, "w": 0, "theta_x": 1, "theta_y": 2}

COLORMAP = plt.get_cmap("turbo").with_extremes(over="lightgrey", under="lightgrey")
GRAY_COLORMAP = mcolors.ListedColormap(["grey"])  #Used when no field is shown

#====================================================================
# Symbols for the boundary conditions, with the node at (0, 0):
#   v prescribed: A triangle with its tip at the node, and a line underneath (a roller)
#   u prescribed: The same, rotated 90 degrees
#   u and v prescribed: A triangle on a hatched ground (a pin)
#   w prescribed: A dot
#   thetax and thetay prescribed: An arrow with a double arrowhead along the x- or y-axis. This is the moment
#                                 vector notation for a rotation about that axis (right-hand rule).
#   w, thetax and thetay prescribed: A square (clamped)
# Each symbol has a size (relative to the symbol size) and a label for the legend.
#====================================================================
M, L = Path.MOVETO, Path.LINETO  #Move the pen, or draw a line
BC_SYMBOLS = {
    "v": (Path([(0, 0), (-0.6, -0.8), (0.6, -0.8), (0, 0), (-0.6, -1), (0.6, -1)],
               [M, L, L, L, M, L]), 2.8, r"$v$ prescribed"),
    "u": (Path([(0, 0), (-0.8, 0.6), (-0.8, -0.6), (0, 0), (-1, 0.6), (-1, -0.6)],
               [M, L, L, L, M, L]), 2.8, r"$u$ prescribed"),
    "u and v": (Path([(0, 0), (-0.6, -0.75), (0.6, -0.75), (0, 0), (-0.9, -0.75), (0.9, -0.75), (-0.9, -1),
                      (-0.65, -0.75), (-0.45, -1), (-0.2, -0.75), (0, -1), (0.25, -0.75), (0.45, -1), (0.7, -0.75)],
                     [M, L, L, L, M, L, M, L, M, L, M, L, M, L]), 2.8, r"$u$ and $v$ prescribed"),
    "w": ("o", 0.9, r"$w$ prescribed"),
    "theta_x": (Path([(-1, 0), (1, 0), (0.65, 0.3), (1, 0), (0.65, -0.3), (0.3, 0.3), (0.65, 0), (0.3, -0.3)],
                     [M, L, M, L, L, M, L, L]), 2.2, r"$\theta_x$ prescribed"),
    "theta_y": (Path([(0, -1), (0, 1), (0.3, 0.65), (0, 1), (-0.3, 0.65), (0.3, 0.3), (0, 0.65), (-0.3, 0.3)],
                     [M, L, M, L, L, M, L, L]), 2.2, r"$\theta_y$ prescribed"),
    "clamped": ("s", 0.9, r"$w$, $\theta_x$ and $\theta_y$ prescribed (clamped)"),
}


def get_field_names(problem_type):
    """The fields that can be plotted for each problem type"""
    if problem_type == ProblemType.PLANE_STRESS:
        return ["none", "u", "v", "u_mag", "sigma_xx", "sigma_yy", "sigma_xy", "sigma_vm"]
    else:
        assert problem_type == ProblemType.PLATE
        return ["none", "w", "theta_x", "theta_y", "M_xx", "M_yy", "M_xy", "V_x", "V_y", "applied_load"]


def is_stress_field(field):
    return field in ["sigma_xx", "sigma_yy", "sigma_xy", "sigma_vm", "M_xx", "M_yy", "M_xy", "V_x", "V_y"]


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
    outline = ([point(i, 0) for i in range(n)] + [point(n, j) for j in range(n)] + [point(n - i, n) for i in range(n)] +
               [point(0, n - j) for j in range(n + 1)])
    return xi_eta, np.array(triangles), np.array(outline)


def calc_applied_load_in_element(config: Config, mesh: Mesh, e, N):
    """The load per area on a plate from all the loads on the face of element e, at the points where the shape
    functions N are given. The load functions are evaluated at the points. A pressure counts as it is, and a body
    force counts as the body force times the thickness, like in the load integration. (Line loads on the edges are
    not included.)"""
    nodeIDs = mesh.elements[e, :]
    x = N @ mesh.nodes[nodeIDs, 0]
    y = N @ mesh.nodes[nodeIDs, 1]
    load_per_area = np.zeros(len(N))
    for load in config.loads:
        element_set = mesh.element_sets[load.element_set_name]
        if element_set.element_entity != ElementEntity.FACE or e not in element_set.elementIDs:
            continue
        for i in range(len(N)):
            value = loads.evaluate_scalar_load(load, x[i], y[i])
            if load.load_type == LoadType.BODY_FORCE:
                value *= config.h
            load_per_area[i] += value
    return load_per_area


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
    if field == "applied_load":
        return calc_applied_load_in_element(config, mesh, e, N)

    #The rest of the fields are stresses
    if config.stress_recovery == "consistent":
        stress = stress_recovery.calc_element_stress(config, mesh, solver_data.r, e, N, dNdxi, dNdeta)
    else:
        assert config.stress_recovery == "L2"
        stress = N @ nodal_stress_L2[nodeIDs]  #Interpolate the nodal stresses with the shape functions
    if field == "sigma_vm":
        return stress_recovery.calc_von_mises(stress)
    return stress[:, stress_recovery.get_stress_names(config.problem_type).index(field)]


def calc_disp_scaling(config: Config, mesh: Mesh, solver_data: SolverData):
    """The factor that the displacements are multiplied by in the plots. If config.disp_scaling is "auto", the
    largest displacement is shown as 10% of the size of the mesh (rounded to two digits)."""
    if config.disp_scaling != "auto":
        return config.disp_scaling
    u, v = unpack_solution(config, solver_data.r)
    largest_displacement = np.max(np.sqrt(u**2 + v**2))
    if not largest_displacement > 0:  #No displacements, or NaN from a singular system
        return 1.0
    mesh_size = max(np.ptp(mesh.nodes[:, 0]), np.ptp(mesh.nodes[:, 1]))
    return float(f"{0.1 * mesh_size / largest_displacement:.2g}")


def get_element_size(mesh: Mesh):
    """A typical element size: the shortest of the four sides of each element, as a median over all elements.
    This works for any quad mesh, also unstructured ones."""
    corners = mesh.nodes[mesh.elements[:, :4]]  #The four corners of each element, one row per element
    shortest_side = np.full(mesh.get_nE(), np.inf)
    for corner_a, corner_b in QUAD_EDGES:
        side = np.linalg.norm(corners[:, corner_b] - corners[:, corner_a], axis=1)
        shortest_side = np.minimum(shortest_side, side)
    return np.median(shortest_side)


def get_screen_length_of_unit_length(ax):
    """The length on the screen (in points, 1/72 inch) of one unit of length in the plot"""
    ax.autoscale_view()
    ax.apply_aspect()
    x0 = ax.transData.transform((0, 0))[0]
    x1 = ax.transData.transform((1, 0))[0]
    return (x1 - x0) * 72 / ax.figure.dpi


def calc_element_edge_width(ax, mesh: Mesh):
    """The line width (in points) of the element edges. Small elements on the screen get thinner lines,
    so that a fine mesh doesn't hide the contours."""
    element_size_on_screen = get_element_size(mesh) * get_screen_length_of_unit_length(ax)
    return np.clip(0.15 * element_size_on_screen, 0.1, 1.0)


def plot_field(ax, config: Config, mesh: Mesh, solver_data: SolverData, field, nodal_stress_L2=None):
    """Draws a field (e.g. "u_mag" or "sigma_xx", see get_field_names) as a contour plot on the mesh, in the
    matplotlib axes ax. For plane stress, the mesh is deformed by the displacements times config.disp_scaling
    (see calc_disp_scaling), unless config.show_deformed is False.
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
    deformed = config.problem_type == ProblemType.PLANE_STRESS and config.show_deformed
    if deformed:
        disp_scaling = calc_disp_scaling(config, mesh, solver_data)
    for e in range(nE):
        nodeIDs = mesh.elements[e, :]
        x[e] = N @ mesh.nodes[nodeIDs, 0]
        y[e] = N @ mesh.nodes[nodeIDs, 1]
        if deformed:
            x[e] += disp_scaling * (N @ r[nodeIDs, 0])
            y[e] += disp_scaling * (N @ r[nodeIDs, 1])
        values[e] = calc_field_in_element(config, mesh, solver_data, field, e, N, dNdxi, dNdeta, nodal_stress_L2)

    #The plot points of element e are numbered from e * (number of plot points per element)
    first_point_of_element = len(xi_eta) * np.arange(nE)
    all_triangles = (triangles[None, :, :] + first_point_of_element[:, None, None]).reshape(-1, 3)
    triangulation = tri.Triangulation(x.ravel(), y.ravel(), all_triangles)
    if field == "none":
        contours = ax.tripcolor(triangulation, values.ravel(), shading="gouraud", cmap=GRAY_COLORMAP)
    else:
        contours = ax.tripcolor(triangulation, values.ravel(), shading="gouraud", cmap=COLORMAP)

    ax.set_aspect("equal")
    if config.show_element_edges:
        element_outlines = [np.column_stack([x[e, outline], y[e, outline]]) for e in range(nE)]
        element_edges = LineCollection(element_outlines, colors="black", gid="element_edges")
        ax.add_collection(element_edges)
        element_edges.set_linewidth(calc_element_edge_width(ax, mesh))
    return contours


#====================================================================
# The GUI
#====================================================================


def get_node_spacing(mesh: Mesh, element_type):
    """A typical distance between neighbouring nodes, used to choose the size of the nodes and labels"""
    return get_element_size(mesh) / (ELEMENT_TRAITS[element_type].nNl_1D - 1)


def calc_outward_normals(mesh: Mesh, element_type):
    """Returns an outward unit normal for each node on the boundary of the mesh, and NaN for the other nodes.
    The boundary consists of the element edges that belong to only one element. The normal of a node is the
    average of the normals of the boundary edges that it lies on (e.g. diagonal at a corner)."""
    elements = mesh.elements
    nodes = mesh.nodes

    #Count how many elements each edge belongs to. An edge is given by its two corner nodes (sorted).
    edge_count = {}
    for e in range(mesh.get_nE()):
        for corner_a, corner_b in QUAD_EDGES:
            edge = tuple(sorted((elements[e, corner_a], elements[e, corner_b])))
            edge_count[edge] = edge_count.get(edge, 0) + 1

    normals = np.zeros((mesh.get_nN(), 2))
    for e in range(mesh.get_nE()):
        for edge_id, (corner_a, corner_b) in enumerate(QUAD_EDGES):
            I_a, I_b = elements[e, corner_a], elements[e, corner_b]
            if edge_count[tuple(sorted((I_a, I_b)))] == 1:  #The edge is on the boundary
                #The corners go counter-clockwise, so the outward normal is the edge turned 90 degrees clockwise
                dx, dy = nodes[I_b] - nodes[I_a]
                edge_normal = np.array([dy, -dx]) / np.sqrt(dx**2 + dy**2)
                for il in ELEMENT_TRAITS[element_type].entity_nodes[ElementEntity(edge_id)]:
                    normals[elements[e, il]] += edge_normal

    length = np.sqrt(normals[:, 0]**2 + normals[:, 1]**2)
    on_boundary = length > 0
    normals[on_boundary] /= length[on_boundary, None]
    normals[~on_boundary] = np.nan
    return normals


def format_contour_limit(value):
    return f"{value:.3e}"


def show_text_in_box(text_box, text):
    """Changes the text that a text box shows. (TextBox.set_val would also trigger update() again, show the cursor as
    if the user was typing, and redraw the whole window.)"""
    text_box.text_disp.set_text(text)


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
        self.outward_normals = calc_outward_normals(mesh, config.element_type)  #Used to draw forces outside the body
        self.has_center_node = (0, 0) in ELEMENT_TRAITS[config.element_type].xi_eta  #True for Q9

        self.field_names = get_field_names(config.problem_type)
        if config.contour not in self.field_names:
            raise ValueError(f"Unknown config.contour '{config.contour}' for {config.problem_type.name}. "
                             f"Available: {', '.join(self.field_names)}")

        if config.problem_type == ProblemType.PLATE and config.arrow_type != "none":
            raise ValueError("Arrows are only available for plane stress problems. Use config.arrow_type = 'none'.")

        self.fig, self.ax = plt.subplots(figsize=(11, 8))
        plt.subplots_adjust(left=0.3, bottom=0.25)
        self.create_widgets()
        if config.problem_type == ProblemType.PLANE_STRESS:
            problem_name = "Plane stress"
        else:
            assert config.problem_type == ProblemType.PLATE
            problem_name = "Mindlin plate"
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
        self.radio_contour = RadioButtons(get_widget_axes(0.03, 0.2, 0.22),
                                          self.contour_labels,
                                          active=self.field_names.index(config.contour))
        self.radio_contour.ax.set_title("Contour:")

        recovery_options = ["consistent", "L2"]
        self.radio_recovery = RadioButtons(get_widget_axes(0.03, 0.2, 0.05),
                                           recovery_options,
                                           active=recovery_options.index(config.stress_recovery))
        self.radio_recovery.ax.set_title("Stress recovery:")

        #Arrows are only shown for plane stress. (For plates, the loads and reactions are out of the plane.)
        self.radio_arrows = None
        if config.problem_type == ProblemType.PLANE_STRESS:
            arrow_options = ["none", "disp", "external forces", "reaction forces"]
            self.radio_arrows = RadioButtons(get_widget_axes(0.03, 0.2, 0.08),
                                             arrow_options,
                                             active=arrow_options.index(config.arrow_type))
            self.radio_arrows.ax.set_title("Arrows:")

        show_options = ["labels", "nodes", "element edges", "boundary conditions"]
        show_status = [config.show_labels, config.show_nodes, config.show_element_edges, config.show_bcs]
        if config.problem_type == ProblemType.PLANE_STRESS:  #Plates are drawn undeformed
            show_options.append("deformed shape")
            show_status.append(config.show_deformed)
        self.check_buttons = CheckButtons(get_widget_axes(0.03, 0.2, 0.02 * len(show_options)), show_options,
                                          show_status)
        self.check_buttons.ax.set_title("Show:")

        #The disp scale box is only shown when "deformed shape" is ticked (see update)
        self.box_disp_scale = None
        if config.problem_type == ProblemType.PLANE_STRESS:
            self.box_disp_scale = TextBox(get_widget_axes(0.13, 0.08, 0.03), "disp scale ", initial=config.disp_scaling)

        #The contour limit boxes are only shown when "specify contour limits" is ticked (see update)
        self.check_specify_limits = CheckButtons(get_widget_axes(0.03, 0.2, 0.03), ["specify contour limits"],
                                                 [config.specify_contour_limits])
        #The min and max boxes are placed next to each other
        bottom_of_row = self.top_of_next_widget - 0.03
        self.box_contour_min = TextBox(plt.axes([0.06, bottom_of_row, 0.065, 0.03]), "min ", initial=config.contour_min)
        self.box_contour_max = TextBox(plt.axes([0.165, bottom_of_row, 0.065, 0.03]),
                                       "max ",
                                       initial=config.contour_max)

        #The widgets at the bottom
        self.slider_symbol_scale = Slider(plt.axes([0.4, 0.13, 0.45, 0.03]),
                                          "symbol size",
                                          0,
                                          5,
                                          valinit=config.symbol_scale)
        self.slider_arrow_scale = None
        if self.radio_arrows is not None:  #Only for plane stress, like the arrows
            self.slider_arrow_scale = Slider(plt.axes([0.4, 0.08, 0.45, 0.03]),
                                             "arrow size",
                                             0,
                                             5,
                                             valinit=config.arrow_scale)
        self.save_button = Button(plt.axes([0.4, 0.02, 0.15, 0.04]),
                                  "Save Screen",
                                  color="lightblue",
                                  hovercolor="skyblue")
        self.save_button.on_clicked(self.save_screen)

        for radio in [self.radio_contour, self.radio_recovery, self.radio_arrows]:
            if radio is not None:
                radio.on_clicked(self.update)
        self.check_buttons.on_clicked(self.update)
        self.check_specify_limits.on_clicked(self.update)
        self.box_contour_max.on_submit(self.update)
        self.box_contour_min.on_submit(self.update)
        if self.box_disp_scale is not None:
            self.box_disp_scale.on_submit(self.update)
        self.slider_symbol_scale.on_changed(self.update)
        if self.slider_arrow_scale is not None:
            self.slider_arrow_scale.on_changed(self.update)

        #All the text in the widgets, so that the font size can be changed when the window changes size
        widgets = [
            self.radio_contour, self.radio_recovery, self.radio_arrows, self.check_buttons, self.box_disp_scale,
            self.check_specify_limits, self.box_contour_max, self.box_contour_min, self.slider_symbol_scale,
            self.slider_arrow_scale, self.save_button
        ]
        self.widget_texts = [text for widget in widgets if widget is not None for text in widget.ax.findobj(Text)]

    def read_widgets(self):
        """Copies the choices in the GUI to config"""
        config = self.config
        config.contour = self.field_names[self.contour_labels.index(self.radio_contour.value_selected)]
        config.stress_recovery = self.radio_recovery.value_selected
        if self.radio_arrows is not None:
            config.arrow_type = self.radio_arrows.value_selected
        show_status = self.check_buttons.get_status()
        config.show_labels, config.show_nodes, config.show_element_edges, config.show_bcs = show_status[:4]
        if config.problem_type == ProblemType.PLANE_STRESS:
            config.show_deformed = show_status[4]
        config.specify_contour_limits = self.check_specify_limits.get_status()[0]
        config.symbol_scale = self.slider_symbol_scale.val
        if self.slider_arrow_scale is not None:
            config.arrow_scale = self.slider_arrow_scale.val
        if self.box_disp_scale is not None:
            if self.box_disp_scale.text.strip().lower() == "auto":
                config.disp_scaling = "auto"
            else:
                try:
                    config.disp_scaling = float(self.box_disp_scale.text)
                except ValueError:
                    print(f"Invalid input for the displacement scaling: '{self.box_disp_scale.text}'. Using 'auto'.")
                    config.disp_scaling = "auto"

        #The contour limits are shown rounded in the text boxes. They are only read if the user has changed the
        #text, so that ticking "specify contour limits" doesn't replace the exact limits by the rounded ones.
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

        #Replace "auto" by the displacement scaling it gives, and show it in the text box
        if self.box_disp_scale is not None:
            config.disp_scaling = calc_disp_scaling(config, self.mesh, self.solver_data)
            show_text_in_box(self.box_disp_scale, f"{config.disp_scaling:g}")
            self.box_disp_scale.ax.set_visible(config.show_deformed)

        #====================================================================
        # Contour plot. The L2 stresses are only calculated once.
        #====================================================================
        if is_stress_field(config.contour) and config.stress_recovery == "L2" and self.nodal_stress_L2 is None:
            self.nodal_stress_L2 = stress_recovery.calc_nodal_stress_L2(config, self.mesh, self.solver_data.r)
        contours = plot_field(self.ax, config, self.mesh, self.solver_data, config.contour, self.nodal_stress_L2)
        self.update_contour_limits_and_colorbar(contours)

        #The stress recovery only matters for stresses, so it is greyed out when another field is shown
        if is_stress_field(config.contour):
            recovery_color = "black"
        else:
            recovery_color = "lightgrey"
        for text in list(self.radio_recovery.labels) + [self.radio_recovery.ax.title]:
            text.set_color(recovery_color)

        #The boxes for the contour limits are only shown when they are specified (otherwise, see the colorbar)
        self.box_contour_max.ax.set_visible(config.specify_contour_limits)
        self.box_contour_min.ax.set_visible(config.specify_contour_limits)

        #====================================================================
        # Nodes, arrows and boundary conditions
        #====================================================================
        if config.problem_type == ProblemType.PLANE_STRESS and config.show_deformed:
            u, v = unpack_solution(config, self.solver_data.r)
            x_nodes = self.mesh.nodes[:, 0] + config.disp_scaling * u
            y_nodes = self.mesh.nodes[:, 1] + config.disp_scaling * v
        else:
            x_nodes = self.mesh.nodes[:, 0]
            y_nodes = self.mesh.nodes[:, 1]
        self.legend_handles = []  #The arrows and boundary conditions add their symbols to the legend
        self.plot_nodes_and_labels(x_nodes, y_nodes)
        self.plot_arrow_field(x_nodes, y_nodes)
        self.plot_bcs(x_nodes, y_nodes)
        if len(self.legend_handles) > 0:  #The legend is placed above the plot, so that it doesn't cover anything
            self.ax.legend(handles=self.legend_handles,
                           loc="lower left",
                           bbox_to_anchor=(0, 1.01),
                           borderaxespad=0,
                           ncol=min(len(self.legend_handles), 2))

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

        if config.specify_contour_limits:
            contours.set_clim(config.contour_min, config.contour_max)
        else:
            #Remember the limits, and show them in the text boxes
            config.contour_min, config.contour_max = contours.get_clim()
            show_text_in_box(self.box_contour_min, format_contour_limit(config.contour_min))
            show_text_in_box(self.box_contour_max, format_contour_limit(config.contour_max))

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
        size = self.get_symbol_size() * self.config.symbol_scale
        if self.node_markers is not None:
            #The nodes may become much smaller than the other symbols, so that they don't cover a fine mesh
            node_size = self.get_symbol_size(smallest=0.5) * self.config.symbol_scale
            self.node_markers.set_sizes([(0.5 * node_size)**2])
        for label in self.node_labels:
            label.set_fontsize(size)
            label.xyann = (0.25 * size, 0.25 * size)  #Distance from the node (in points)
        for label in self.element_labels:
            label.set_fontsize(size)
            if self.has_center_node:
                label.xyann = (0, -0.4 * size)  #Below the center node (in points)
        for markers, relative_size in self.bc_markers:
            markers.set_sizes([(relative_size * size)**2])
            markers.set_linewidths(max(0.8, 0.12 * size))
        for collection in self.ax.collections:  #The element edges get thinner when the elements get small
            if collection.get_gid() == "element_edges":
                collection.set_linewidth(calc_element_edge_width(self.ax, self.mesh))

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
                    self.ax.annotate(str(i), (x_nodes[i], y_nodes[i]),
                                     xytext=(0, 0),
                                     textcoords="offset points",
                                     zorder=20))
            #The element numbers are shown in parentheses at the center of the elements (xi = eta = 0). If there is
            #a node at the center (Q9), the number is placed just below it instead (see apply_sizes).
            N_center = shape_functions.calc_N(0.0, 0.0, config.element_type)
            if self.has_center_node:
                vertical_alignment = "top"
            else:
                vertical_alignment = "center"
            for e in range(self.mesh.get_nE()):
                nodeIDs = self.mesh.elements[e, :]
                center = (N_center @ x_nodes[nodeIDs], N_center @ y_nodes[nodeIDs])
                self.element_labels.append(
                    self.ax.annotate(f"({e})",
                                     center,
                                     xytext=(0, 0),
                                     textcoords="offset points",
                                     ha="center",
                                     va=vertical_alignment,
                                     zorder=20))

    def plot_bcs(self, x_nodes, y_nodes):
        """Draws a symbol at each node with a boundary condition (see BC_SYMBOLS), and adds them to the legend"""
        self.bc_markers = []  #Pairs of (markers, size relative to the symbol size), used in apply_sizes
        if not self.config.show_bcs:
            return

        #Find the prescribed dofs of each node, and which symbols the nodes get
        prescribed_dofs = {}
        for bc in self.config.bcs:
            for I in self.mesh.node_sets[bc.node_set_name]:
                prescribed_dofs.setdefault(I, set()).add(bc.dof)
        nodes_with_symbol = {symbol_name: [] for symbol_name in BC_SYMBOLS}
        for I, dofs in prescribed_dofs.items():
            if Dof.U in dofs and Dof.V in dofs:
                nodes_with_symbol["u and v"].append(I)
            elif Dof.U in dofs:
                nodes_with_symbol["u"].append(I)
            elif Dof.V in dofs:
                nodes_with_symbol["v"].append(I)
            if Dof.W in dofs and Dof.THETAX in dofs and Dof.THETAY in dofs:
                nodes_with_symbol["clamped"].append(I)
            else:
                for dof, symbol_name in [(Dof.W, "w"), (Dof.THETAX, "theta_x"), (Dof.THETAY, "theta_y")]:
                    if dof in dofs:
                        nodes_with_symbol[symbol_name].append(I)

        for symbol_name, nodeIDs in nodes_with_symbol.items():
            if len(nodeIDs) == 0:
                continue
            marker, relative_size, label = BC_SYMBOLS[symbol_name]
            if symbol_name in ["w", "clamped"]:  #The dots and squares are filled, and drawn behind the nodes
                fill_color = "magenta"
                zorder = 9
            else:
                fill_color = "none"
                zorder = 15
            markers = self.ax.scatter(x_nodes[nodeIDs],
                                      y_nodes[nodeIDs],
                                      marker=marker,
                                      facecolors=fill_color,
                                      edgecolors="magenta",
                                      zorder=zorder)
            self.bc_markers.append((markers, relative_size))
            self.legend_handles.append(
                mlines.Line2D([], [],
                              marker=marker,
                              markersize=12 * min(relative_size, 1.2),
                              linestyle="None",
                              markerfacecolor=fill_color,
                              markeredgecolor="magenta",
                              label=label))

        #Make room for the symbols at the edges of the plot, so that they are not cut off. A symbol reaches half
        #its size (the largest relative size is 2.8) from the node, which is converted to the units of the plot.
        reach_on_screen = 0.5 * 2.8 * self.get_symbol_size() * self.config.symbol_scale
        reach = reach_on_screen / get_screen_length_of_unit_length(self.ax)
        nodeIDs = list(prescribed_dofs.keys())
        node_positions = np.column_stack([x_nodes[nodeIDs], y_nodes[nodeIDs]])
        self.ax.update_datalim(np.vstack([node_positions - reach, node_positions + reach]))

    def get_symbol_size(self, smallest=3):
        """The size (in points, 1/72 inch) that the nodes, labels, boundary condition symbols and arrows are based on,
        before the size sliders are applied. It follows the distance between the nodes on the screen, but is kept
        between a smallest and a largest size (which follow the window size, like the fonts)."""
        window_scale = self.fig.get_size_inches()[1] / self.DESIGN_HEIGHT
        node_spacing_on_screen = self.node_spacing * get_screen_length_of_unit_length(self.ax)
        return np.clip(0.35 * node_spacing_on_screen, smallest * window_scale, 12 * window_scale)

    def plot_arrow_field(self, x_nodes, y_nodes):
        config = self.config
        if config.arrow_type == "none":
            return
        if config.arrow_type == "disp":
            field_x, field_y = unpack_solution(config, self.solver_data.r)
            self.plot_arrows(x_nodes, y_nodes, field_x, field_y, "disp", "blue", draw_outside=False)
        elif config.arrow_type == "external forces":
            field_x, field_y = unpack_solution(config, self.solver_data.R_ext)
            self.plot_arrows(x_nodes, y_nodes, field_x, field_y, "external forces", "red", draw_outside=True)
        elif config.arrow_type == "reaction forces":
            field_x, field_y = unpack_solution(config, self.solver_data.R_int - self.solver_data.R_ext)
            self.plot_arrows(x_nodes, y_nodes, field_x, field_y, "reaction forces", "green", draw_outside=True)
        else:
            raise ValueError(f"Unknown arrow type for plotting: {config.arrow_type}")

    def plot_arrows(self, x, y, field_x, field_y, label, color, draw_outside):
        """Draws the vectors (field_x, field_y) as arrows at the points (x, y). The longest arrow is 5 symbol sizes
        long, but at least 25 points (times the arrow size slider). The other arrows are scaled in proportion to it.
        If draw_outside is True, arrows at the boundary that point into the body are drawn outside it, with the tip
        at the node, so that they don't cover the plot (like forces are usually drawn)."""
        magnitude = np.sqrt(field_x**2 + field_y**2)
        if not np.any(magnitude > 0):  #Nothing to draw, e.g. no external forces, or NaN from a singular system
            return
        largest = np.max(magnitude[np.isfinite(magnitude)])
        keep = magnitude > 1e-6 * largest  #Skip the arrows that are too small to see (e.g. round-off errors)

        window_scale = self.fig.get_size_inches()[1] / self.DESIGN_HEIGHT
        longest_arrow_on_screen = max(5 * self.get_symbol_size(), 25 * window_scale) * self.config.arrow_scale
        #The arrows are given in the units of the plot, so the length on the screen is converted to those units
        longest_arrow = longest_arrow_on_screen / get_screen_length_of_unit_length(self.ax)
        arrow_x = field_x[keep] * longest_arrow / largest
        arrow_y = field_y[keep] * longest_arrow / largest

        start_x = x[keep]
        start_y = y[keep]
        if draw_outside:
            normals = self.outward_normals[keep]  #NaN for nodes that are not on the boundary
            points_into_body = arrow_x * normals[:, 0] + arrow_y * normals[:, 1] < 0
            start_x = np.where(points_into_body, start_x - arrow_x, start_x)  #Then the tip ends at the node
            start_y = np.where(points_into_body, start_y - arrow_y, start_y)

        self.ax.quiver(
            start_x,
            start_y,
            arrow_x,
            arrow_y,
            color=color,
            zorder=16,  #On top of the support symbols
            angles="xy",
            scale_units="xy",
            scale=1,
            units="xy",
            width=0.05 * longest_arrow)
        #Make room for arrows that point out of the plot
        self.ax.update_datalim(np.column_stack([start_x, start_y]))
        self.ax.update_datalim(np.column_stack([start_x + arrow_x, start_y + arrow_y]))

        #The arrows are scaled, so the legend shows the size of the largest one
        self.legend_handles.append(
            mlines.Line2D([], [],
                          color=color,
                          marker=r'$\rightarrow$',
                          linestyle='None',
                          markersize=10,
                          label=f"{label} (largest: {largest:.2e})"))
