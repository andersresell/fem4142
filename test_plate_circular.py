from fem.useful_imports import *  #import required functions

if __name__ == "__main__":

    E = 200000e6
    nu = 0.3
    R = 1.0
    Lx = R
    Ly = R
    q = -1e6
    problem_type = ProblemType.PLATE
    nEx = 1  # 4
    nEy = 1  # 4
    thickness_ratios = np.logspace(1, 6, 10)
    # thickness_ratios = [10, 20, 30, 50, 100, 200, 300, 500, 1000, 1e6]
    # thickness_ratios = [10, 100, 1000, 10000]

    clamped = False

    plt.figure()

    element_types = [ElementType.Q4, ElementType.Q4R, ElementType.Q8R, ElementType.Q8, ElementType.Q9, ElementType.Q16]

    element_types = [ElementType.Q16]
    for element_type in element_types:
        wc_over_wc_k = []
        for tr in thickness_ratios:
            h = R / tr
            config = create_config(E, nu, h, element_type, problem_type)
            mesh = create_structured_quad_mesh(config, Lx, Ly, nEx, nEy)
            morph_mesh_to_quarter_circle(mesh)

            #model one quartrer of the plate
            add_boundary_condition(config, mesh, "west", Dof.THETAY, 0)
            add_boundary_condition(config, mesh, "south", Dof.THETAX, 0)

            #Clamp the real boundaries
            add_boundary_condition(config, mesh, "east", Dof.W, 0)
            if clamped:
                add_boundary_condition(config, mesh, "east", Dof.THETAX, 0)
                add_boundary_condition(config, mesh, "east", Dof.THETAY, 0)
            add_boundary_condition(config, mesh, "north", Dof.W, 0)
            if clamped:
                add_boundary_condition(config, mesh, "north", Dof.THETAX, 0)
                add_boundary_condition(config, mesh, "north", Dof.THETAY, 0)

            load_func = lambda x, y: q * x
            add_load(config, mesh, "domain", LoadType.PRESSURE, load_func)

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
            #  assert np.isclose(w_max, wc), f"Max deflection {w_max} not at center but {wc}"

            print(f"Element type: {element_type.name}, R/h: {tr:.1f}, wc: {wc:.6e} m")

            D = E * h**3 / (12 * (1 - nu**2))
            if clamped:
                wc_k = 1.0 / 64 * abs(q) * R**4 / D
            else:
                wc_k = (5 + nu) / (64 * (1 + nu)) * abs(q) * R**4 / D

            wc_over_wc_k.append(wc / wc_k)

            Plot(config, mesh, solver_data)
        plt.plot(thickness_ratios, wc_over_wc_k, '.-', label=element_type.name)
    if clamped:
        plt.title(
            f"Clamped circular plate, center deflection over Kirchhoff solution\n{nEx}x{nEy} elements per quarter plate"
        )
    else:
        plt.title(
            f"Simply supported circular plate,\ncenter deflection over Kirchhoff solution\n{nEx}x{nEy} elements per quarter plate"
        )

    #log plot
    plt.xscale('log')
    plt.xlabel("R/h")
    plt.ylabel("wc / wc_k")
    #plt.ylim([0.7, 1.2])
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()
