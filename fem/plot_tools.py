import os
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.collections import LineCollection
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


def get_mesh_bounds(nodes):
    x_min = np.min(nodes[:, 0])
    x_max = np.max(nodes[:, 0])
    y_min = np.min(nodes[:, 1])
    y_max = np.max(nodes[:, 1])
    return x_min, x_max, y_min, y_max


def get_L_domain(nodes):
    x_min, x_max, y_min, y_max = get_mesh_bounds(nodes)
    return max(x_max - x_min, y_max - y_min)


class Plot:
    """Class for plotting 2D finite element meshes and results. 
    Includes some GUI features for interactive plotting.
    Simply construct an instance of the class to start the GUI."""

    # Font size settings for GUI
    LABEL_FONTSIZE = 20  # Axis labels and colorbar labels
    TICK_FONTSIZE = 16  # Tick labels on axes and colorbar

    gray_val = 0.5  # 0=black, 1=white
    cmap_name_gray = mcolors.LinearSegmentedColormap.from_list("uniform_gray", [(gray_val, gray_val, gray_val)] * 2)
    cmap_name_default = "turbo"
    # cmap_default = "plasma"
    plt.rcParams.update({
        'font.size': 11  # or any desired size
    })

    def __init__(self, config: Config, mesh: Mesh, solver_data: SolverData):
        self.set_num_quads_1D_per_element_for_reasonable_plotting(mesh.get_nE())
        self.fig, self.ax = plt.subplots(figsize=(12, 10), dpi=100)
        self.cbar = None
        self.config = config
        self.mesh = mesh
        self.solver_data = solver_data

        plt.subplots_adjust(left=0.3, bottom=0.25)

        VSPACE_BETWEEN_WIDGETS = 0.04
        self.curr_vspace_left = 0.9

        def get_vspace_left(height):
            vspace = self.curr_vspace_left - height / 2
            assert self.curr_vspace_left > 0.1
            self.curr_vspace_left -= (height + VSPACE_BETWEEN_WIDGETS)
            return vspace

        self.curr_vspace_bottom_index = 0
        vspaces_bottom = np.linspace(0.15, 0.05, 4)

        def get_new_vspace_bottom():
            assert self.curr_vspace_bottom_index < len(vspaces_bottom)
            vspace = vspaces_bottom[self.curr_vspace_bottom_index]
            self.curr_vspace_bottom_index += 1
            return vspace

        #Problem type specific options:
        if config.problem_type == ProblemType.PLANE_STRESS:
            contour_type_options = ['none', 'disp', 'stress']
            disp_component_options = ['mag', 'u', 'v']
            stress_component_options = ['mises', 'xx', 'yy', 'xy']
            arrow_options = ['none', 'disp', 'external forces', 'reaction forces']
        else:
            assert config.problem_type == ProblemType.PLATE
            #Stress plotting is not implemented for plates yet, so that option is not shown
            contour_type_options = ['none', 'disp']
            disp_component_options = ['w', 'thetax', 'thetay']
            stress_component_options = ['mises', 'xx', 'yy', 'xy']
            arrow_options = ['none']
            if config.contour_type == 'stress':
                raise ValueError("Stress contours are not implemented for plate problems yet. "
                                 "Use config.contour_type = 'disp' instead.")

        radio_contour_type = RadioButtons(plt.axes([0.05, get_vspace_left(0.05), 0.15, 0.05]),
                                          contour_type_options,
                                          active=contour_type_options.index(config.contour_type))
        radio_contour_type.ax.set_title('Contour type:')

        radio_disp_component = RadioButtons(plt.axes([0.05, get_vspace_left(0.06), 0.15, 0.06]),
                                            disp_component_options,
                                            active=disp_component_options.index(config.disp_component))
        radio_disp_component.ax.set_title('Disp component:')
        radio_stress_component = RadioButtons(plt.axes([0.05, get_vspace_left(0.07), 0.15, 0.07]),
                                              stress_component_options,
                                              active=stress_component_options.index(config.stress_component))
        radio_stress_component.ax.set_title('Stress component:')
        radio_arrows = RadioButtons(plt.axes([0.05, get_vspace_left(0.07), 0.15, 0.07]),
                                    arrow_options,
                                    active=arrow_options.index(config.arrow_type))
        radio_arrows.ax.set_title('Arrow type:')

        check_buttons = CheckButtons(
            plt.axes([0.05, get_vspace_left(0.07), 0.15,
                      0.07]), ['show node labels', 'specify contour limits', 'show mesh', 'show boundary conditions'],
            [config.show_node_labels, config.specify_contour_limits, config.show_mesh, config.show_bcs])
        check_buttons.ax.set_title('Options:')

        self.box_contour_max = TextBox(plt.axes([0.1, get_vspace_left(0.03), 0.07, 0.03]),
                                       'contour lim max ',
                                       initial=config.contour_max)
        self.box_contour_min = TextBox(plt.axes([0.1, get_vspace_left(0.03), 0.07, 0.03]),
                                       'contour lim min ',
                                       initial=config.contour_min)

        box_disp_scale = TextBox(plt.axes([0.1, get_vspace_left(0.03), 0.04, 0.03]),
                                 'disp scale ',
                                 initial=config.disp_scaling)

        slider_node_scale = Slider(plt.axes([0.25, get_new_vspace_bottom(), 0.50, 0.03]),
                                   'node size',
                                   0,
                                   10,
                                   valinit=config.node_scale)
        slider_arrow_scale = Slider(plt.axes([0.25, get_new_vspace_bottom(), 0.50, 0.03]),
                                    'arrow size',
                                    0,
                                    10,
                                    valinit=config.arrow_scale)

        self.save_button = Button(plt.axes([0.25, get_new_vspace_bottom(), 0.15, 0.03]),
                                  'Save Screen',
                                  color='lightblue',
                                  hovercolor='skyblue')
        self.save_button.on_clicked(self.save_screen)

        def update(val):
            try:
                config.disp_scaling = float(box_disp_scale.text)
            except ValueError:
                print("Invalid input for displacement scaling. Using default value.")
                config.disp_scaling = 1.0

            config.contour_type = radio_contour_type.value_selected
            config.disp_component = radio_disp_component.value_selected
            config.stress_component = radio_stress_component.value_selected
            config.arrow_type = radio_arrows.value_selected
            config.node_scale = slider_node_scale.val
            config.arrow_scale = slider_arrow_scale.val
            config.show_node_labels = check_buttons.get_status()[0]
            config.specify_contour_limits = check_buttons.get_status()[1]
            config.show_mesh = check_buttons.get_status()[2]
            config.show_bcs = check_buttons.get_status()[3]

            try:
                config.contour_min = float(self.box_contour_min.text)
            except ValueError:
                print("Invalid input for contour min. Using default value.")
                config.contour_min = 0.0

            try:
                config.contour_max = float(self.box_contour_max.text)
            except ValueError:
                print("Invalid input for contour max. Using default value.")
                config.contour_max = 1.0

            self.ax.cla()
            self.plot_2d(config, mesh, solver_data)
            self.fig.canvas.draw_idle()

        box_disp_scale.on_submit(update)
        slider_node_scale.on_changed(update)
        slider_arrow_scale.on_changed(update)
        radio_contour_type.on_clicked(update)
        radio_disp_component.on_clicked(update)
        radio_stress_component.on_clicked(update)
        radio_arrows.on_clicked(update)
        check_buttons.on_clicked(update)
        self.box_contour_max.on_submit(update)
        self.box_contour_min.on_submit(update)

        update(None)  # initial draw
        plt.show()

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

    def plot_2d(self, config: Config, mesh: Mesh, solver_data: SolverData):

        x_vert, y_vert, scalar_vert, triangles, outlines = self.create_render_mesh(mesh.get_nE())
        nN = mesh.get_nN()
        #====================================================================
        # Calculate position of vertices, displacements, etc.
        #====================================================================
        self.set_vertex_pos_and_color(x_vert, y_vert, scalar_vert, mesh, solver_data, config)

        #====================================================================
        # Create triangulation of the mesh for plotting and find max/min
        # values for the color map
        #====================================================================
        tris = tri.Triangulation(x_vert, y_vert, triangles)
        cmap_name = self.cmap_name_default
        if config.contour_type == "none":
            vmin = 0
            vmax = 1
            cmap_name = self.cmap_name_gray
        elif config.specify_contour_limits:
            vmin = config.contour_min
            vmax = config.contour_max
        else:
            vmin = np.min(scalar_vert)
            vmax = np.max(scalar_vert)
            config.contour_min = vmin
            config.contour_max = vmax
            #Show the limits in the text boxes. The events must be switched off while doing this, since
            #changing the text otherwise triggers update(), which would draw the whole plot again.
            for text_box, value in [(self.box_contour_min, vmin), (self.box_contour_max, vmax)]:
                text_box.eventson = False
                text_box.set_val(f"{value:.3e}")
                text_box.eventson = True

        cmap = plt.get_cmap(cmap_name).with_extremes(over='lightgrey', under='lightgrey')

        #====================================================================
        # Plot the mesh body with the specified contour type
        #====================================================================
        tpc = self.ax.tripcolor(tris, scalar_vert, shading='gouraud', cmap=cmap, vmin=vmin, vmax=vmax)

        fig = self.ax.get_figure()

        #====================================================================
        # Colorbar
        #====================================================================
        if self.cbar is None:
            self.cbar = fig.colorbar(tpc, ax=self.ax)
        self.cbar.ax.set_visible(True)
        self.cbar.update_normal(tpc)
        vmin, vmax = tpc.get_clim()

        # Force uniformly spaced colorbar ticks
        NUM_TICKS = 10
        ticks = np.linspace(vmin, vmax, NUM_TICKS)
        self.cbar.set_ticks(ticks)

        # Set colorbar tick label font size
        self.cbar.ax.tick_params(labelsize=self.TICK_FONTSIZE)

        #====================================================================
        # Set axis labels and colorbar label
        #====================================================================
        if config.contour_type == "none":
            self.cbar.ax.set_visible(False)
        elif config.problem_type == ProblemType.PLANE_STRESS:
            if config.contour_type == "disp":
                if config.disp_component == "mag":
                    self.cbar.set_label(r"$u^{\mathrm{mag}} = \sqrt{u^2 + v^2}$", fontsize=self.LABEL_FONTSIZE)
                elif config.disp_component == "u":
                    self.cbar.set_label(r"$u$", fontsize=self.LABEL_FONTSIZE)
                elif config.disp_component == "v":
                    self.cbar.set_label(r"$v$", fontsize=self.LABEL_FONTSIZE)
                else:
                    print(f"Error: Unknown displacement component for plotting: {config.disp_component}")
            elif config.contour_type == "stress":
                if config.stress_component == "mises":
                    self.cbar.set_label(r"$\sigma^{\mathrm{VM}}$", fontsize=self.LABEL_FONTSIZE)
                elif config.stress_component == "xx":
                    self.cbar.set_label(r"$\sigma_{\mathrm{xx}}$", fontsize=self.LABEL_FONTSIZE)
                elif config.stress_component == "yy":
                    self.cbar.set_label(r"$\sigma_{\mathrm{yy}}$", fontsize=self.LABEL_FONTSIZE)
                elif config.stress_component == "xy":
                    self.cbar.set_label(r"$\sigma_{\mathrm{xy}}$", fontsize=self.LABEL_FONTSIZE)
                else:
                    print(f"Error: Unknown stress component for plotting: {config.stress_component}")
            else:
                print(f"Error: Unknown contour type for plotting: {config.contour_type}")
        else:
            assert config.problem_type == ProblemType.PLATE
            if config.contour_type == "disp":
                if config.disp_component == "w":
                    self.cbar.set_label(r"$w$", fontsize=self.LABEL_FONTSIZE)
                elif config.disp_component == "thetax":
                    self.cbar.set_label(r"$\theta_x$", fontsize=self.LABEL_FONTSIZE)
                elif config.disp_component == "thetay":
                    self.cbar.set_label(r"$\theta_y$", fontsize=self.LABEL_FONTSIZE)
                else:
                    print(f"Error: Unknown displacement component for plotting: {config.disp_component}")
            elif config.contour_type == "stress":
                print("Stress contour plotting for plate problems is not implemented yet.")
                assert False
            else:
                print(f"Error: Unknown contour type for plotting: {config.contour_type}")

        #====================================================================
        # Plot lines marking edges of elements
        #====================================================================
        self.plot_element_outlines(config, x_vert, y_vert, outlines)

        #====================================================================
        # Plot nodes
        #====================================================================
        SCALE_NODES = 1
        if config.problem_type == ProblemType.PLANE_STRESS:
            u, v = unpack_solution(config, solver_data.r)
            x_nodes = mesh.nodes[:, 0] + config.disp_scaling * u
            y_nodes = mesh.nodes[:, 1] + config.disp_scaling * v
        else:
            assert config.problem_type == ProblemType.PLATE
            x_nodes = mesh.nodes[:, 0]
            y_nodes = mesh.nodes[:, 1]

        SCALE_DOMAIN = 20
        domain_scale = SCALE_DOMAIN / np.sqrt(nN)
        if config.show_mesh:
            self.ax.scatter(x_nodes,
                            y_nodes,
                            color="black",
                            s=config.node_scale * domain_scale * SCALE_NODES,
                            zorder=10)
        SCALE_NODELABELS = 3
        if config.show_node_labels:
            for i in range(nN):
                self.ax.text(x_nodes[i],
                             y_nodes[i],
                             str(i),
                             fontsize=config.node_scale * domain_scale * SCALE_NODELABELS,
                             color="black",
                             ha="left",
                             va="bottom",
                             zorder=20)

        #====================================================================
        # Plot arrow fields
        #====================================================================
        if config.arrow_type != "none":
            if config.problem_type == ProblemType.PLANE_STRESS:
                show_max_norm = False
                if config.arrow_type == "disp":
                    self.plot_arrows(x_nodes, y_nodes, u, v, "disp", "blue", domain_scale, config.arrow_scale)
                elif config.arrow_type == "external forces":
                    Rx_ext, Ry_ext = unpack_solution(config, solver_data.R_ext)
                    self.plot_arrows(x_nodes, y_nodes, Rx_ext, Ry_ext, "external forces", "red", domain_scale,
                                     config.arrow_scale, show_max_norm)
                elif config.arrow_type == "reaction forces":
                    R_rea = solver_data.R_int - solver_data.R_ext
                    Rx_rea, Ry_rea = unpack_solution(config, R_rea)
                    self.plot_arrows(x_nodes, y_nodes, Rx_rea, Ry_rea, "reaction forces", "green", domain_scale,
                                     config.arrow_scale, show_max_norm)
                else:
                    raise ValueError(f"Unknown arrow type for plotting: {config.arrow_type}")
            else:
                assert False, "Arrows are not implemented for plate problems"

        #====================================================================
        # Plot boundary conditions
        #====================================================================
        self.plot_bcs(config, mesh, x_nodes, y_nodes, domain_scale)

        if config.problem_type == ProblemType.PLANE_STRESS:
            fig.suptitle(f"Plane stress problem: {config.element_type.name} element")
        else:
            assert config.problem_type == ProblemType.PLATE
            fig.suptitle(f"Mindlin plate problem: {config.element_type.name} element")
        self.ax.set_aspect('equal')
        self.ax.axis('equal')
        self.ax.set_xlabel(r"$x$", fontsize=self.LABEL_FONTSIZE)
        self.ax.set_ylabel(r"$y$", fontsize=self.LABEL_FONTSIZE)

        # Set axis tick label font size
        self.ax.tick_params(axis='both', which='major', labelsize=self.TICK_FONTSIZE)

    def plot_arrows(self, x, y, field_x, field_y, label, color, domain_scale, user_scale, show_max_norm=False):
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
        #handle normalization in a non singular way
        field_x_normalized = field_x_keep / (max_vector_norm + 1e-10)  #ensure safe division
        field_y_normalized = field_y_keep / (max_vector_norm + 1e-10)  #ensure safe division

        self.ax.quiver(
            x_keep,
            y_keep,
            ARROW_LENGTH_SCALE * user_scale * field_x_normalized,
            ARROW_LENGTH_SCALE * user_scale * field_y_normalized,
            color=color,
            scale=1,  #this scale thing is weird, increasing it makes arrows smaller
            width=ARROW_WIDTH_SCALE * domain_scale)

        # Create a proxy handle for the legend
        if show_max_norm:
            label += f" (max norm: {max_vector_norm:.2e})"
        arrow_proxy = mlines.Line2D([], [],
                                    color=color,
                                    marker=r'$\rightarrow$',
                                    linestyle='None',
                                    markersize=10,
                                    label=label)
        self.ax.legend(handles=[arrow_proxy])

    def IX_viz_E(self, e, i, j):
        num_quads_1D = self.NUM_QUADS_PLOT_1D_PER_ELEMENT
        curr_size_tris = e * (num_quads_1D + 1)**2
        return curr_size_tris + i * (num_quads_1D + 1) + j

    def xi_eta_viz(self, i, j):
        num_quads_1D = self.NUM_QUADS_PLOT_1D_PER_ELEMENT
        xi = -1 + 2 * i / num_quads_1D
        eta = -1 + 2 * j / num_quads_1D
        assert xi >= -1 and xi <= 1
        assert eta >= -1 and eta <= 1
        return xi, eta

    def create_render_mesh(self, nE):
        """Triangulates the mesh for plotting. Each quad element is subdivided into num_quads_1D x num_quads_1D smaller quads
        where each smaller quad is divided into two triangles. The positions and colors will be interpolated by the element shape functions"""
        num_quads_1D = self.NUM_QUADS_PLOT_1D_PER_ELEMENT

        triangles_glob = []
        outline_glob = []
        x_vert = np.zeros(nE * (num_quads_1D + 1)**2)
        y_vert = np.zeros_like(x_vert)
        color_vert = np.zeros_like(x_vert)

        for e in range(nE):
            #====================================================================
            # The quads making up the element body
            #====================================================================
            for i in range(num_quads_1D):
                for j in range(num_quads_1D):
                    v0 = self.IX_viz_E(e, i, j)
                    v1 = self.IX_viz_E(e, i + 1, j)
                    v2 = self.IX_viz_E(e, i + 1, j + 1)
                    v3 = self.IX_viz_E(e, i, j + 1)
                    triangles_glob.append([v0, v1, v2])
                    triangles_glob.append([v0, v2, v3])

            #====================================================================
            # The four outlines of the element
            #====================================================================

            #Bottom line:
            for i in range(num_quads_1D):
                v0 = self.IX_viz_E(e, i, 0)
                v1 = self.IX_viz_E(e, i + 1, 0)
                outline_glob.append([v0, v1])
            #Right line:
            for j in range(num_quads_1D):
                v0 = self.IX_viz_E(e, num_quads_1D, j)
                v1 = self.IX_viz_E(e, num_quads_1D, j + 1)
                outline_glob.append([v0, v1])
            #Top line:
            for i in range(num_quads_1D):
                v0 = self.IX_viz_E(e, num_quads_1D - i, num_quads_1D)
                v1 = self.IX_viz_E(e, num_quads_1D - i - 1, num_quads_1D)
                outline_glob.append([v0, v1])
            #Left line:
            for j in range(num_quads_1D):
                v0 = self.IX_viz_E(e, 0, num_quads_1D - j)
                v1 = self.IX_viz_E(e, 0, num_quads_1D - j - 1)
                outline_glob.append([v0, v1])

        return x_vert, y_vert, color_vert, np.array(triangles_glob, dtype=int), np.array(outline_glob, dtype=int)

    def set_vertex_pos_and_color(self, x_vert, y_vert, scalar_vert, mesh: Mesh, solver_data: SolverData,
                                 config: Config):
        num_quads_1D = self.NUM_QUADS_PLOT_1D_PER_ELEMENT
        nE = mesh.get_nE()
        nodes = mesh.nodes
        elements = mesh.elements
        element_type = config.element_type
        if config.problem_type == ProblemType.PLANE_STRESS:
            u, v = unpack_solution(config, solver_data.r)

            for e in range(nE):
                for i in range(num_quads_1D + 1):
                    for j in range(num_quads_1D + 1):
                        xi, eta = self.xi_eta_viz(i, j)
                        N = shape_functions.calc_N(xi, eta, element_type)
                        x_e = nodes[elements[e, :], 0]
                        y_e = nodes[elements[e, :], 1]
                        u_e = u[elements[e, :]]
                        v_e = v[elements[e, :]]
                        x_viz = np.dot(N, x_e + u_e * config.disp_scaling)
                        y_viz = np.dot(N, y_e + v_e * config.disp_scaling)
                        scalar = 0.0
                        if config.contour_type == "disp":
                            if config.disp_component == "mag":
                                scalar = np.sqrt((np.dot(N, u_e))**2 + (np.dot(N, v_e))**2)
                            elif config.disp_component == "u":
                                scalar = np.dot(N, u_e)
                            else:
                                assert config.disp_component == "v"
                                scalar = np.dot(N, v_e)

                        #Set the vertex values
                        I_vert = self.IX_viz_E(e, i, j)
                        x_vert[I_vert] = x_viz
                        y_vert[I_vert] = y_viz
                        scalar_vert[I_vert] = scalar

            if config.contour_type == "stress":
                xi_eta = [self.xi_eta_viz(i, j) for i in range(num_quads_1D + 1) for j in range(num_quads_1D + 1)]
                for e in range(nE):
                    sigma_E = stress_recovery.recover_stress_plane_stress_consistent(e, config, mesh, nodes, u, v, xi_eta)
                    for i in range(num_quads_1D + 1):
                        for j in range(num_quads_1D + 1):
                            I_loc = self.IX_viz_E(0, i, j)  #relative index
                            sigma = sigma_E[I_loc, :]
                            sxx = sigma[0]
                            syy = sigma[1]
                            sxy = sigma[2]
                            if config.stress_component == "mises":
                                val = np.sqrt(sxx**2 - sxx * syy + syy**2 + 3 * sxy**2)
                            elif config.stress_component == "xx":
                                val = sxx
                            elif config.stress_component == "yy":
                                val = syy
                            else:
                                assert config.stress_component == "xy"
                                val = sxy
                            I_vert = self.IX_viz_E(e, i, j)
                            scalar_vert[I_vert] = val

        else:
            assert config.problem_type == ProblemType.PLATE
            w, thetax, thetay = unpack_solution(config, solver_data.r)
            for e in range(nE):
                for i in range(num_quads_1D + 1):
                    for j in range(num_quads_1D + 1):
                        xi, eta = self.xi_eta_viz(i, j)
                        N = shape_functions.calc_N(xi, eta, element_type)
                        x_e = nodes[elements[e, :], 0]
                        y_e = nodes[elements[e, :], 1]
                        w_e = w[elements[e, :]]
                        thetax_e = thetax[elements[e, :]]
                        thetay_e = thetay[elements[e, :]]
                        x_viz = np.dot(N, x_e)
                        y_viz = np.dot(N, y_e)
                        scalar = 0.0
                        if config.contour_type == "disp":
                            if config.disp_component == "w":
                                scalar = np.dot(N, w_e)
                            elif config.disp_component == "thetax":
                                scalar = np.dot(N, thetax_e)
                            else:
                                assert config.disp_component == "thetay"
                                scalar = np.dot(N, thetay_e)

                        #Set the vertex values
                        I_vert = self.IX_viz_E(e, i, j)
                        x_vert[I_vert] = x_viz
                        y_vert[I_vert] = y_viz
                        scalar_vert[I_vert] = scalar
            assert config.contour_type != "stress", "Stress contours are not implemented for plate problems yet"

    def plot_element_outlines(self, config: Config, x_vert, y_vert, outlines):
        if not config.show_mesh:
            return
        segments = [[[x_vert[a], y_vert[a]], [x_vert[b], y_vert[b]]] for a, b in outlines]
        line_collection = LineCollection(segments, colors='black', linewidths=1.0)
        self.ax.add_collection(line_collection)

    def plot_bcs(self, config: Config, mesh: Mesh, x_nodes, y_nodes, domain_scale):
        #====================================================================
        # Plot boundary conditions
        #====================================================================
        def get_segments_of_symbol(dof, s, x0, y0):
            segs = []
            FACTOR = 0.8
            if dof == Dof.V:
                sx = s * FACTOR
                sy = s
                #A triangle with a line underneath. Center is at top vertex of triangle
                segs.append([[-sx, -sy], [sx, -sy], [0, 0], [-sx, -sy]])  #triangle
                segs.append([[-sx, -1.5 * sy], [sx, -1.5 * sy]])  #line underneath
            elif dof == Dof.U:
                sx = s
                sy = s * FACTOR
                #Same as V, but rotated -90 degrees,
                segs.append([[-sx, sy], [-sx, -sy], [0, 0], [-sx, sy]])  #triangle
                segs.append([[-1.5 * sx, -sy], [-1.5 * sx, sy]])
            else:
                assert False
            for seg in segs:
                for coords in seg:
                    coords[0] += x0
                    coords[1] += y0
            return segs

        if config.show_bcs:
            L_domain = get_L_domain(mesh.nodes)
            s = min(L_domain / 40, L_domain * 0.01 * domain_scale)
            if config.problem_type == ProblemType.PLANE_STRESS:
                marker_size = 1.3
            else:
                assert config.problem_type == ProblemType.PLATE
                marker_size = min(7, 5 * domain_scale)
            segments = []
            x_w_constrained = []
            y_w_constrained = []
            for bc in config.bcs:
                nodeIDs_constrained = mesh.node_sets[bc.node_set_name]
                for I in nodeIDs_constrained:
                    x0, y0 = x_nodes[I], y_nodes[I]
                    if bc.dof == Dof.U or bc.dof == Dof.V:
                        segments.extend(get_segments_of_symbol(bc.dof, s, x0, y0))
                    elif bc.dof == Dof.THETAX:
                        segments.append([[x0, y0 - s], [x0, y0 + s]])  #vertical line
                    elif bc.dof == Dof.THETAY:
                        segments.append([[x0 - s, y0], [x0 + s, y0]])  #horizontal line
                    else:
                        assert config.problem_type == ProblemType.PLATE and bc.dof == Dof.W
                        x_w_constrained.append(x0)
                        y_w_constrained.append(y0)

            bc_lines = LineCollection(segments, colors='magenta', linewidths=marker_size)
            self.ax.add_collection(bc_lines)
            #Plot a dot for each node with constrained w
            self.ax.plot(x_w_constrained, y_w_constrained, "o", color='magenta', markersize=marker_size)

    def set_num_quads_1D_per_element_for_reasonable_plotting(self, nE):
        MAX_1D = 40
        MIN_1D = 5
        # TARGET = 2000
        TARGET = 4000
        num_quads_1D = int(np.sqrt(TARGET / nE))
        num_quads_1D = int(np.clip(num_quads_1D, MIN_1D, MAX_1D))
        assert MIN_1D <= num_quads_1D <= MAX_1D
        self.NUM_QUADS_PLOT_1D_PER_ELEMENT = num_quads_1D
        print(f"Using {num_quads_1D}x{num_quads_1D} quads per element for contour plotting.")
