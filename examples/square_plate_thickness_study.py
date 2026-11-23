import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  #Lets Python find the fem folder
from fem.useful_imports import *  #import required functions

#====================================================================
# Shear locking of Mindlin plate elements: A clamped square plate under
# uniform pressure is solved for thinner and thinner plates (larger L/h).
# The center deflection is divided by the Kirchhoff (thin plate)
# solution, so a good element approaches 1 for thin plates. An element
# that locks becomes far too stiff, and its curve drops towards 0.
# For thick plates (small L/h), the ratio is above 1, since Mindlin
# theory includes shear deformation, which Kirchhoff theory neglects.
#====================================================================

if __name__ == "__main__":

    E = 200000e6  # Young's modulus in Pa
    nu = 0.3  # Poisson's ratio
    L = 1.0  #Side length of the plate
    Lx = L / 2  #Only one quarter of the plate is modelled, using symmetry
    Ly = L / 2
    q = -1e6  #Pressure. Negative, so that the plate deflects upwards and the largest w is at the center
    problem_type = ProblemType.PLATE
    nEx = 4  #Number of elements in x-direction (per quarter plate)
    nEy = 4  #Number of elements in y-direction (per quarter plate)
    plot_and_interrupt = False  #Set to True to show the GUI for each solution
    thickness_ratios = np.logspace(1, 6, 15)  #L/h, from thick to very thin plates

    plt.figure()

    element_types = [ElementType.Q4R, ElementType.Q4, ElementType.Q8, ElementType.Q16]
    for element_type in element_types:
        wc_over_wc_k = []
        for tr in thickness_ratios:
            h = L / tr
            config = create_config(E, nu, h, element_type, problem_type)
            mesh = create_structured_quad_mesh(config, Lx, Ly, nEx, nEy)

            #model one quarter of the plate
            add_boundary_condition(config, mesh, "west", Dof.THETAY, 0)
            add_boundary_condition(config, mesh, "south", Dof.THETAX, 0)

            #Clamp the real boundaries
            add_boundary_condition(config, mesh, "east", Dof.W, 0)
            add_boundary_condition(config, mesh, "east", Dof.THETAX, 0)
            add_boundary_condition(config, mesh, "east", Dof.THETAY, 0)
            add_boundary_condition(config, mesh, "north", Dof.W, 0)
            add_boundary_condition(config, mesh, "north", Dof.THETAX, 0)
            add_boundary_condition(config, mesh, "north", Dof.THETAY, 0)

            load_func = lambda x, y: q
            add_load(config, mesh, "domain", LoadType.PRESSURE, load_func)

            solver_data = create_solver_data(config, mesh)
            solve(config, solver_data, mesh)

            w, _, _ = unpack_solution(config, solver_data.r)
            w_max = np.max(w)
            nodes = mesh.nodes
            x_nodes = nodes[:, 0]
            y_nodes = nodes[:, 1]
            #Check that max deflection is at center
            dist_to_center = np.sqrt(x_nodes**2 + y_nodes**2)
            center_node_index = np.argmin(dist_to_center)
            wc = w[center_node_index]
            assert np.isclose(w_max, wc), f"Max deflection {w_max} not at center but {wc}"

            print(f"Element type: {element_type.name}, L/h: {tr:.1f}, wc: {wc:.6e} m")

            D = E * h**3 / (12 * (1 - nu**2))
            wc_k = 0.00126 * abs(q) * L**4 / D  #Kirchhoff solution for a clamped square plate (Timoshenko)
            wc_over_wc_k.append(wc / wc_k)
            if plot_and_interrupt:
                Plot(config, mesh, solver_data)
        plt.plot(thickness_ratios, wc_over_wc_k, '.-', label=element_type.name)

    plt.title(f"Clamped plate, center deflection over Kirchhoff solution\n{nEx}x{nEy} elements per quarter plate")

    #log plot
    plt.xscale('log')
    plt.xlabel("L/h")
    plt.ylabel("wc / wc_k")
    plt.ylim([0, 1.2])
    plt.grid()
    plt.legend()
    plt.show()
