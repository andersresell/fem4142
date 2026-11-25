import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  #Lets Python find the fem folder
from fem.useful_imports import *  #import required functions

#====================================================================
# A thick-walled pipe with internal pressure, compared with Lamé's
# exact solution along the x-axis. There, sigma_xx and sigma_yy are
# the radial and hoop stresses sigma_r and sigma_theta. One quarter
# of the pipe is modelled, using symmetry.
#
# Plane stress is used, so strictly speaking this is a thin ring. For
# this load case, sigma_r and sigma_theta are the same as in a long
# pipe (plane strain), but the displacements are not.
#====================================================================

if __name__ == "__main__":

    E = 210e9  # Young's modulus in Pa
    nu = 0.3  # Poisson's ratio
    h = 1.0  #Thickness out of the plane in m
    a = 0.1  #Inner radius in m
    b = 0.2  #Outer radius in m (b/a = 2 is a thick wall)
    p = 10e6  #Internal pressure in Pa (100 bar)
    element_type = ElementType.Q4  #Try also ElementType.Q4, whose straight edges cut the corners of the arcs
    problem_type = ProblemType.PLANE_STRESS
    nEx = 3  #Number of elements through the wall
    nEy = 8  #Number of elements around the quarter circle

    #====================================================================
    # Create a mesh of a quarter ring. "west" is the inner arc, "east"
    # the outer arc, "south" lies along the x-axis and "north" along
    # the y-axis.
    #====================================================================
    config = create_config(E, nu, h, element_type, problem_type)
    mesh = create_structured_quad_mesh(config, 1.0, 1.0, nEx, nEy)  #The size doesn't matter, the morphing rescales it
    morph_mesh_to_quarter_annulus(mesh, inner_radius=a, outer_radius=b)

    #====================================================================
    # Symmetry conditions: the nodes on the x-axis can only move along x,
    # and the nodes on the y-axis can only move along y
    #====================================================================
    add_boundary_condition(config, mesh, "south", Dof.V, 0)
    add_boundary_condition(config, mesh, "north", Dof.U, 0)

    #====================================================================
    # The internal pressure acts on the inner arc
    #====================================================================
    load_func = lambda x, y: p
    add_load(config, mesh, "west", LoadType.PRESSURE, load_func)

    solver_data = create_solver_data(config, mesh)
    solve(config, solver_data, mesh)

    #====================================================================
    # Take out the stresses along the x-axis (the node set "south"),
    # sorted by x. The stresses are found with L2 stress recovery.
    #====================================================================
    nodal_stress = stress_recovery.calc_nodal_stress_L2(config, mesh,
                                                        solver_data.r)  #Columns: sigma_xx, sigma_yy, sigma_xy
    nodeIDs = mesh.node_sets["south"]
    nodeIDs_ordered = nodeIDs[np.argsort(mesh.nodes[nodeIDs, 0])]
    r = mesh.nodes[nodeIDs_ordered, 0]
    sigma_r = nodal_stress[nodeIDs_ordered, 0]
    sigma_theta = nodal_stress[nodeIDs_ordered, 1]

    #====================================================================
    # Lamé's solution for a thick-walled cylinder with internal pressure
    #====================================================================
    A = p * a**2 / (b**2 - a**2)
    B = p * a**2 * b**2 / (b**2 - a**2)
    r_theory = np.linspace(a, b, 100)
    sigma_r_theory = A - B / r_theory**2
    sigma_theta_theory = A + B / r_theory**2

    #====================================================================
    # The thin-walled formula sigma_theta = p * r / t assumes that the
    # hoop stress is the same through the whole wall. Here, the largest
    # hoop stress (at the inner wall) is clearly higher.
    #====================================================================
    wall_thickness = b - a
    mean_radius = (a + b) / 2
    sigma_theta_thin_wall = p * mean_radius / wall_thickness
    print(f"Largest hoop stress, FEM:                       {sigma_theta[0] / 1e6:.2f} MPa")
    print(f"Largest hoop stress, Lamé:                      {(A + B / a**2) / 1e6:.2f} MPa")
    print(f"Hoop stress, thin-walled formula (mean radius): {sigma_theta_thin_wall / 1e6:.2f} MPa")

    #The stresses are divided by p, so that sigma_r goes from -1 at the inner wall to 0 at the outer wall
    plt.figure()
    plt.plot(r, sigma_r / p, 'o', label="sigma_xx (FEM, L2)")
    plt.plot(r_theory, sigma_r_theory / p, '-', label="sigma_r (Lamé)")
    plt.plot(r, sigma_theta / p, 's', label="sigma_yy (FEM, L2)")
    plt.plot(r_theory, sigma_theta_theory / p, '--', label="sigma_theta (Lamé)")
    plt.axhline(sigma_theta_thin_wall / p, color="gray", linestyle=":", label="sigma_theta (thin-walled formula)")
    plt.xlabel("r")
    plt.ylabel("Stress / p")
    plt.title(f"Thick-walled pipe with internal pressure, {element_type.name} elements")
    plt.legend()
    plt.tight_layout()

    #====================================================================
    # Start the GUI. The von Mises stress only depends on r, so the
    # contours should form rings.
    #====================================================================
    config.contour = "sigma_vm"
    config.arrow_type = "external forces"
    Plot(config, mesh, solver_data)
