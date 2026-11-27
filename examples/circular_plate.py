import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  #Lets Python find the fem folder
from fem.useful_imports import *  #import required functions

#====================================================================
# A circular plate under uniform pressure, compared with Kirchhoff
# (thin plate) theory along the x-axis: The deflection w, and the
# bending moments M_xx and M_yy, which are the radial and tangential
# moments M_r and M_theta along the x-axis. One quarter of the plate
# is modelled, using symmetry.
#====================================================================

if __name__ == "__main__":

    E = 210e9  # Young's modulus in Pa
    nu = 0.3  # Poisson's ratio
    R = 1.0  #Radius of the plate
    h = R / 100  #Plate thickness (thin, so that Kirchhoff theory applies)
    p = 1000  #Pressure in Pa
    clamped = False  #True: clamped edge, False: simply supported edge
    element_type = ElementType.Q16  #Try also ElementType.Q9, whose moments are much less accurate
    problem_type = ProblemType.PLATE
    nEx = 4  #Number of elements in x-direction (per quarter plate)
    nEy = 4  #Number of elements in y-direction (per quarter plate)

    #====================================================================
    # Create a mesh of a quarter circle. The edges "east" and "north"
    # together form the arc.
    #====================================================================
    config = create_config(E, nu, h, element_type, problem_type)
    mesh = create_structured_quad_mesh(config, R, R, nEx, nEy)
    morph_mesh_to_quarter_circle(mesh, radius=R)

    #====================================================================
    # Symmetry conditions on the straight edges, and supports on the arc
    #====================================================================
    add_boundary_condition(config, mesh, "west", Dof.THETAY, 0)
    add_boundary_condition(config, mesh, "south", Dof.THETAX, 0)
    for edge in ["east", "north"]:
        add_boundary_condition(config, mesh, edge, Dof.W, 0)
        if clamped:
            add_boundary_condition(config, mesh, edge, Dof.THETAX, 0)
            add_boundary_condition(config, mesh, edge, Dof.THETAY, 0)

    load_func = lambda x, y: p
    add_load(config, mesh, "domain", LoadType.PRESSURE, load_func)

    solver_data = create_solver_data(config, mesh)
    solve(config, solver_data, mesh)

    #====================================================================
    # Take out w and the moments along the x-axis (the node set "south"),
    # sorted by x. The moments are found with L2 stress recovery.
    #====================================================================
    w, thetax, thetay = unpack_solution(config, solver_data.r)
    nodal_moments = stress_recovery.calc_nodal_stress_L2(config, mesh,
                                                         solver_data.r)  #Columns: M_xx, M_yy, M_xy, V_x, V_y
    nodeIDs = mesh.node_sets["south"]
    nodeIDs_ordered = nodeIDs[np.argsort(mesh.nodes[nodeIDs, 0])]
    r = mesh.nodes[nodeIDs_ordered, 0]

    #====================================================================
    # Kirchhoff solution (Timoshenko & Woinowsky-Krieger, Theory of plates
    # and shells). The deflection is positive downwards, and a positive
    # moment gives tension at the bottom.
    #====================================================================
    D = E * h**3 / (12 * (1 - nu**2))
    if clamped:
        w_theory = p * (R**2 - r**2)**2 / (64 * D)
        M_r_theory = p / 16 * (R**2 * (1 + nu) - r**2 * (3 + nu))
        M_theta_theory = p / 16 * (R**2 * (1 + nu) - r**2 * (1 + 3 * nu))
    else:
        w_theory = p * (R**2 - r**2) / (64 * D) * ((5 + nu) / (1 + nu) * R**2 - r**2)
        M_r_theory = p * (3 + nu) / 16 * (R**2 - r**2)
        M_theta_theory = p / 16 * (R**2 * (3 + nu) - r**2 * (1 + 3 * nu))

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(r, -w[nodeIDs_ordered], 'o', label="FEM")  #The pressure pushes the plate downwards, so w < 0
    plt.plot(r, w_theory, '-', label="Kirchhoff theory")
    plt.ylabel("-w")
    if clamped:
        plt.title(f"Clamped circular plate, {element_type.name} elements")
    else:
        plt.title(f"Simply supported circular plate, {element_type.name} elements")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(r, nodal_moments[nodeIDs_ordered, 0], 'o', label="M_xx (FEM)")
    plt.plot(r, M_r_theory, '-', label="M_r (Kirchhoff theory)")
    plt.plot(r, nodal_moments[nodeIDs_ordered, 1], 's', label="M_yy (FEM)")
    plt.plot(r, M_theta_theory, '--', label="M_theta (Kirchhoff theory)")
    plt.xlabel("r")
    plt.ylabel("Moment per unit length")
    plt.legend()
    plt.tight_layout()

    #====================================================================
    # Start the GUI to visualize the results
    #====================================================================
    GUI(config, mesh, solver_data)
