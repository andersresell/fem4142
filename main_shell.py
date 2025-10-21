from src.useful_imports import *  #import required functions

if __name__ == "__main__":

    E = 200000e6
    nu = 0.3
    L = 1.0
    Lx = L / 2
    Ly = L / 2
    q = -1e6
    problem_type = PROBLEM_TYPE_PLATE
    nEx = 2
    nEy = 2
    # thickness_ratios = np.linspace(10, 1e6, 5)
    # thickness_ratios = [10, 20, 30, 50, 100, 200, 300, 500, 1000, 1e6]
    thickness_ratios = np.logspace(1, 6, 15)
    # thickness_ratios = [10]

    plt.figure()

    element_types = [ELEMENT_TYPE_Q4, ELEMENT_TYPE_Q8R, ELEMENT_TYPE_Q8, ELEMENT_TYPE_Q9, ELEMENT_TYPE_Q16]
    # element_types = [ELEMENT_TYPE_Q4]
    for element_type in element_types:
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

            # Plot(config, mesh, solver_data)
        plt.plot(thickness_ratios, wc_over_wc_k, '.-', label=element_type_to_str[element_type])

    plt.title(f"Clamped plate, center deflection over Kirchhoff solution\n{nEx}x{nEy} elements per quarter plate")

    #log plot
    plt.xscale('log')
    plt.xlabel("L/h")
    plt.ylabel("wc / wc_k")
    plt.ylim([0.7, 1.2])
    plt.grid()
    plt.legend()
    plt.show()
