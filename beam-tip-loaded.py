from src.useful_imports import *  #import required functions

np.random.seed(0)
if __name__ == "__main__":

    E = 210e9  # Young's modulus in Pa
    nu = 0.3  # Poisson's ratio
    h = 1  # Plate thickness in m
    Lx = 10  #Length in x-direction
    Ly = Lx / 100  #Length in y-direction (beam height)
    P = 1000  #Concentrated tip load in N
    element_type = ELEMENT_TYPE_Q16  #Use quadrilateral element
    problem_type = PROBLEM_TYPE_PLANE_STRESS  #Specify that a plane stress problem is solved
    nEx = 1  #Number of elements in x-direction
    nEy = 1  #Number of elements in y-direction

    # Convert concentrated tip load to uniform vertical traction on the right edge
    # Total force = traction * (edge_height * thickness) = t_y * (Ly * h)
    # Therefore: t_y = P / (Ly * h)
    t_y = -P / (Ly * h)  #Negative for downward direction
    perturb_mesh_nodes_factor = 0

    #====================================================================
    # Group problem settings in an object called config
    #====================================================================
    config = create_config(E, nu, h, element_type, problem_type)
    config.hourglass_scaling = 0.1
    config.stress_reconstruction_method = STRESS_RECONSTRUCTION_EXTRAPOLATED_FULL

    #====================================================================
    # Create a rectangular structured mesh
    #====================================================================
    mesh = create_structured_quad_mesh(config, Lx, Ly, nEx, nEy, perturb_mesh_nodes_factor)

    #====================================================================
    # Add fixed boundary condition  to the left edge called "west"
    #====================================================================
    # add_boundary_condition(config, mesh, "west", DOF_U, 0)
    # add_boundary_condition(config, mesh, "west", DOF_V, 0)

    add_boundary_condition(config, mesh, "west", DOF_U, 0)
    add_boundary_condition(config, mesh, "south_west", DOF_V, 0)

    #====================================================================
    # Apply a uniform vertical traction on the right edge "east" to model tip load
    #====================================================================
    load_func = lambda x, y: (0, t_y)  #(t_x, t_y) - only vertical component
    add_load(config, mesh, "east", LOAD_TYPE_TRACTION, load_func)

    #====================================================================
    # Create objects holding system matrices and vectors. Then assemble
    # the stiffness matrix, integrate loads and assign boundary conditions
    # before finally solving the system K*r = R_ext
    #====================================================================
    solver_data = create_solver_data(config, mesh)
    solve(config, solver_data, mesh)

    #====================================================================
    # Post-processing: Compare FEM solution to analytical beam theory
    # For a cantilever beam with a concentrated tip load P at x=Lx:
    # v(x) = -P/(6*E*I) * (3*Lx*x^2 - x^3)
    # v_tip = v(Lx) = -P*Lx^3 / (3*E*I)
    #====================================================================
    #Fetch the individual displacement components from the generalized displacement vector r
    u, v = unpack_solution(config, solver_data.r)

    # Get displacement along the centerline (y = Ly/2) to compare with beam theory
    # We'll find nodes close to the centerline
    centerline_tolerance = Ly / (2 * nEy)  #Tolerance for finding centerline nodes
    centerline_mask = np.abs(mesh.nodes[:, 1] - Ly / 2) < centerline_tolerance
    nodeIDs_centerline = np.where(centerline_mask)[0]
    x_centerline = mesh.nodes[nodeIDs_centerline, 0]

    # Sort nodes by x-coordinate
    nodeIDs_centerline_ordered = nodeIDs_centerline[np.argsort(x_centerline)]
    x_centerline_ordered = mesh.nodes[nodeIDs_centerline_ordered, 0]
    v_centerline_ordered = v[nodeIDs_centerline_ordered]

    # Calculate second moment of inertia and analytical solution
    I = Ly**3 * h / 12  #Second moment of inertia for a rectangular cross-section

    # Analytical beam deflection for tip-loaded cantilever
    v_theory = -P / (6 * E * I) * (3 * Lx * x_centerline_ordered**2 - x_centerline_ordered**3)
    v_tip_theory = -P * Lx**3 / (3 * E * I)

    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.plot(x_centerline_ordered, v_centerline_ordered, 'o-', label="FEM solution (centerline)", markersize=4)
    plt.plot(x_centerline_ordered, v_theory, '--', label="Beam theory solution", linewidth=2)
    plt.xlabel("x (m)")
    plt.ylabel("v (m)")
    plt.legend()
    plt.title("Cantilever Beam with Tip Load - Deflection Comparison")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Print results
    print("\n" + "=" * 60)
    print("CANTILEVER BEAM WITH TIP LOAD - RESULTS")
    print("=" * 60)
    print(f"Tip load P: {P} N")
    print(f"Beam length Lx: {Lx} m")
    print(f"Beam height Ly: {Ly} m")
    print(f"Thickness h: {h} m")
    print(f"Second moment of inertia I: {I:.6e} m^4")
    print("-" * 60)
    print(f"FEM tip deflection:    {v_centerline_ordered[-1]:.6e} m")
    print(f"Theory tip deflection: {v_tip_theory:.6e} m")
    err_percent = abs((v_centerline_ordered[-1] - v_tip_theory) / v_tip_theory) * 100
    print(f"Relative error: {err_percent:.4f}%")
    print("=" * 60 + "\n")

    #====================================================================
    # Start the GUI to visualize the results
    #====================================================================
    Plot(config, mesh, solver_data)
