import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  #Lets Python find the fem folder
from fem.useful_imports import *  #import required functions

#====================================================================
# Cook's membrane: a classic test of how elements cope with skewed
# meshes. A tapered panel is clamped on the left edge and loaded by a
# shear force on the right edge, so that it bends like a short, deep
# cantilever. The stresses vary a lot: bending, shear, and a stress
# peak in the upper left corner.
#
# The vertical displacement in the middle of the right edge, at
# (48, 52), is compared for different element types and meshes with
# the reference value 23.96 that is commonly used in the literature.
#====================================================================

E = 1.0  # Young's modulus (the benchmark uses these simple numbers)
nu = 1 / 3  # Poisson's ratio
h = 1.0  #Thickness out of the plane
F = 1.0  #Total shear force on the right edge
corners = [(0, 0), (48, 44), (48, 60), (0, 44)]  #South-west, south-east, north-east and north-west corners
v_reference = 23.96  #Vertical displacement in the middle of the right edge


def solve_cooks_membrane(element_type, nE):
    """Solves the problem on a mesh with nE x nE elements (nE must be even, so that there is a node in the middle
    of the right edge), and returns the vertical displacement in the middle of the right edge"""
    config = create_config(E, nu, h, element_type, ProblemType.PLANE_STRESS)
    mesh = create_structured_quad_mesh(config, 1.0, 1.0, nE, nE)  #The size doesn't matter, the morphing rescales it
    morph_mesh_to_quadrilateral(mesh, corners)

    #The left edge is clamped
    add_boundary_condition(config, mesh, "west", Dof.U, 0)
    add_boundary_condition(config, mesh, "west", Dof.V, 0)

    #The shear force acts upwards on the right edge, as a uniform traction
    right_edge_length = corners[2][1] - corners[1][1]
    load_func = lambda x, y: (0, F / (right_edge_length * h))
    add_load(config, mesh, "east", LoadType.TRACTION, load_func)

    solver_data = create_solver_data(config, mesh)
    solve(config, solver_data, mesh)

    #The nodes on the right edge, sorted from the bottom to the top. The middle one is at (48, 52).
    u, v = unpack_solution(config, solver_data.r)
    nodeIDs = mesh.node_sets["east"]
    nodeIDs_ordered = nodeIDs[np.argsort(mesh.nodes[nodeIDs, 1])]
    middle_node = nodeIDs_ordered[len(nodeIDs_ordered) // 2]
    return v[middle_node], config, mesh, solver_data


if __name__ == "__main__":

    element_types = [ElementType.Q4, ElementType.Q4R, ElementType.Q8R, ElementType.Q9, ElementType.Q16]
    nEs = [2, 4, 8, 16]  #Number of elements along each edge

    #====================================================================
    # Solve with each element type on finer and finer meshes
    #====================================================================
    plt.figure()
    for element_type in element_types:
        v_middles = []
        for nE in nEs:
            v_middle, config, mesh, solver_data = solve_cooks_membrane(element_type, nE)
            v_middles.append(v_middle)
        plt.semilogx(nEs, v_middles, 'o-', base=2, label=element_type.name)
        errors = [f"{100 * (v_middle - v_reference) / v_reference:6.1f}%" for v_middle in v_middles]
        print(f"{element_type.name}: error in the displacement on {nEs} meshes: {errors}")

    plt.axhline(v_reference, color='black', linestyle='--', label=f"Reference ({v_reference})")
    plt.xlabel("Number of elements along each edge")
    plt.ylabel("Vertical displacement in the middle of the right edge")
    plt.title("Cook's membrane")
    plt.legend()
    plt.tight_layout()

    #====================================================================
    # Start the GUI for the last solution. The displacement converges
    # as the mesh is refined, but the stress in the upper left corner
    # keeps growing: In theory, it is infinite there, since the clamped
    # edge meets a free edge at a sharp corner.
    #====================================================================
    config.contour = "sigma_vm"
    Plot(config, mesh, solver_data)
