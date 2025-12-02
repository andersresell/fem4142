import sys

sys.path.append('/home/anders/projects/fem4146')

from src.useful_imports import *  #import required functions

if __name__ == "__main__":

    E = 200000e6
    nu = 0.3
    L = 1.0
    Lx = L / 2
    Ly = L / 2
    q = -1e6
    problem_type = PROBLEM_TYPE_PLATE
    nEx = 8
    nEy = 8
    plot_and_interrupt = True
    plot_and_interrupt = False
    # thickness_ratios = np.linspace(10, 1e6, 5)
    # thickness_ratios = [10, 20, 30, 50, 100, 200, 300, 500, 1000, 1e6]
    thickness_ratios = np.logspace(1, 6, 15)
    # thickness_ratios = [10]

    # Set up the plot with consistent style
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 18,
        'axes.titlesize': 20,
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'legend.fontsize': 15,
    })

    # Create figure with better aesthetics
    fig, ax = plt.subplots(figsize=(12, 8))

    # Define colors and markers for each element type
    # Use matplotlib's default color cycle
    colors = [f'C{i}' for i in range(4)]  # C0, C1, C2, C3 (default matplotlib colors)
    markers = ['o', 's', '^', 'd']  # Circle, Square, Triangle, Diamond

    element_types = [
        ELEMENT_TYPE_Q4, ELEMENT_TYPE_Q4R, ELEMENT_TYPE_Q8R, ELEMENT_TYPE_Q8, ELEMENT_TYPE_Q9, ELEMENT_TYPE_Q16
    ]
    element_types = [ELEMENT_TYPE_Q4R, ELEMENT_TYPE_Q4, ELEMENT_TYPE_Q8, ELEMENT_TYPE_Q16]
    #  element_types = [ELEMENT_TYPE_Q9]
    for idx, element_type in enumerate(element_types):
        wc_over_wc_k = []
        for tr in thickness_ratios:
            h = L / tr
            config = create_config(E, nu, h, element_type, problem_type)
            mesh = create_structured_quad_mesh(config, Lx, Ly, nEx, nEy)

            #model one quartrer of the plate
            add_boundary_condition(config, mesh, "west", DOF_THETAY, 0)
            add_boundary_condition(config, mesh, "south", DOF_THETAX, 0)

            #Clamp the real boundaries
            add_boundary_condition(config, mesh, "east", DOF_W, 0)
            add_boundary_condition(config, mesh, "east", DOF_THETAX, 0)
            add_boundary_condition(config, mesh, "east", DOF_THETAY, 0)
            add_boundary_condition(config, mesh, "north", DOF_W, 0)
            add_boundary_condition(config, mesh, "north", DOF_THETAX, 0)
            add_boundary_condition(config, mesh, "north", DOF_THETAY, 0)

            load_func = lambda x, y: q
            add_load(config, mesh, "domain", LOAD_TYPE_PRESSURE, load_func)

            solver_data = create_solver_data(config, mesh)
            solve(config, solver_data, mesh)

            #  Plot(config, mesh, solver_data)

            w, _, _ = unpack_solution(config, solver_data.r)
            w_max = np.max(w)
            nodes = mesh.nodes
            x_nodes = nodes[:, 0]
            y_nodes = nodes[:, 1]
            #Check that max deflection is at center
            #dist_to_center = np.sqrt((x_nodes - Lx / 2)**2 + (y_nodes - Ly / 2)**2)
            dist_to_center = np.sqrt(x_nodes**2 + y_nodes**2)
            center_node_index = np.argmin(dist_to_center)
            wc = w[center_node_index]
            assert np.isclose(w_max, wc), f"Max deflection {w_max} not at center but {wc}"

            print(f"Element type: {element_type_to_str[element_type]}, L/h: {tr:.1f}, wc: {wc:.6e} m")

            D = E * h**3 / (12 * (1 - nu**2))
            wc_k = 0.00126 * abs(q) * L**4 / D
            wc_over_wc_k.append(wc / wc_k)
            if plot_and_interrupt:
                Plot(config, mesh, solver_data)

        # Plot with nice styling
        ax.plot(thickness_ratios,
                wc_over_wc_k,
                marker=markers[idx],
                color=colors[idx],
                linewidth=2.5,
                markersize=8,
                label=element_type_to_str[element_type],
                markeredgecolor='white',
                markeredgewidth=1.0)

    # Reference line at 1.0 (Kirchhoff solution)
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=2.5, alpha=0.7, label='Kirchhoff theory')

    # Styling
    ax.set_xscale('log')
    ax.set_xlabel("Slenderness ratio (L/h)", fontweight='bold')
    ax.set_ylabel(r"$\mathbf{w_c / w_c^{Kirchhoff}}$", fontsize=22)
    ax.set_ylim([0.65, 1.15])
    ax.set_xlim([8, 1.2e6])

    # ax.set_title(f"Shear Locking in Mindlin Plate Elements\nClamped plate, {nEx}×{nEy} elements per quarter",
    #              fontsize=15,
    #              fontweight='bold',
    #              pad=15)

    # Grid - consistent with other exam tasks
    ax.grid(True)
    # Legend
    ax.legend(fontsize=17, loc='lower right', framealpha=0.95, edgecolor='black')

    # Tick parameters
    ax.tick_params(axis='both', which='major', labelsize=15)

    plt.tight_layout()
    plt.savefig('exam-tasks/plate_locking_comparison.png', dpi=600, bbox_inches='tight')
    plt.show()
