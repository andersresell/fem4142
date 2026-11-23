import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  #Lets Python find the fem folder
from fem.useful_imports import *  #import required functions

if __name__ == "__main__":

    E = 210e9  # Young's modulus in Pa
    nu = 0.3  # Poisson's ratio
    h = 1  # Thickness in m
    Lx = 10  #Length in x-direction
    Ly = 1  #Length in y-direction (beam height)
    P = 1000  #Tip load in N
    element_type = ElementType.Q4R  #Q4 with one Gauss point and hourglass stabilization
    problem_type = ProblemType.PLANE_STRESS  #Specify that a plane stress problem is solved
    nEx = 20  #Number of elements in x-direction
    nEy = 4  #Number of elements in y-direction

    #The scaling factors of the hourglass stiffness to compare. The last one is shown in the GUI.
    hourglass_scalings = [10, 1, 0.1, 0.01, 0.001, 0.0001]

    I = Ly**3 * h / 12  #Second moment of inertia for a rectangular cross-section
    v_tip_theory = -P * Lx**3 / (3 * E * I)  #Beam theory solution

    tip_ratios = []
    hourglass_energy_percentages = []
    for hourglass_scaling in hourglass_scalings:
        #====================================================================
        # Solve the cantilever problem with the given hourglass scaling
        #====================================================================
        config = create_config(E, nu, h, element_type, problem_type)
        config.hourglass_scaling = hourglass_scaling
        mesh = create_structured_quad_mesh(config, Lx, Ly, nEx, nEy)
        add_boundary_condition(config, mesh, "west", Dof.U, 0)
        add_boundary_condition(config, mesh, "south_west", Dof.V, 0)
        load_func = lambda x, y: (0, -P / (Ly * h))  #The tip load as a traction on the right edge
        add_load(config, mesh, "east", LoadType.TRACTION, load_func)
        solver_data = create_solver_data(config, mesh)
        solve(config, solver_data, mesh)

        #====================================================================
        # Take out the tip deflection (average of the top and bottom corner)
        # and the part of the strain energy that is in the hourglass stiffness
        #====================================================================
        u, v = unpack_solution(config, solver_data.r)
        v_tip = 0.5 * (v[mesh.node_sets["north_east"][0]] + v[mesh.node_sets["south_east"][0]])
        tip_ratios.append(v_tip / v_tip_theory)
        U_hg, U = calc_hourglass_energy(config, mesh, solver_data)
        hourglass_energy_percentages.append(100 * U_hg / U)

    #====================================================================
    # Too little hourglass stiffness gives hourglass modes (the energy
    # fraction increases), and too much makes the element too stiff
    #====================================================================
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.semilogx(hourglass_scalings, tip_ratios, 'o-')
    plt.axhline(1, color='black', linestyle='--', label="Beam theory")
    plt.ylabel("v_tip / v_tip,theory")
    plt.title(f"Cantilever with {nEx}x{nEy} {element_type.name} elements")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.semilogx(hourglass_scalings, hourglass_energy_percentages, 'o-')
    plt.xlabel("hourglass_scaling")
    plt.ylabel("Hourglass energy (% of total)")
    plt.tight_layout()

    #====================================================================
    # Start the GUI to visualize the results
    #====================================================================
    Plot(config, mesh, solver_data)
