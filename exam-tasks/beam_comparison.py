"""
Beam Comparison Exam Task
Tests a tip-loaded cantilever beam with Q4 elements using three different mesh configurations.
Students should identify which mesh is which based on the results.
"""

import sys

sys.path.append('/home/anders/projects/fem4146')

from src.useful_imports import *
import matplotlib.pyplot as plt
import numpy as np


def run_beam_test(nEx, nEy, Lx=1.0, Ly=0.1):
    """Run a single cantilever beam simulation with tip load."""

    # Material and geometry
    E = 210e9  # Young's modulus in Pa
    nu = 0.3  # Poisson's ratio
    h = 1.0  # Thickness in m
    P = 1000  # Tip load in N

    element_type = ELEMENT_TYPE_Q4
    problem_type = PROBLEM_TYPE_PLANE_STRESS

    # Configuration
    config = create_config(E, nu, h, element_type, problem_type)
    config.show_bcs = True
    config.disp_scaling = 1000
    config.contour_type = "disp"  # Show displacement contours
    config.disp_component = "v"  # Show v (vertical) displacement component
    config.arrow_type = "external forces"  # Display external force arrows

    # Create mesh
    mesh = create_structured_quad_mesh(config, Lx, Ly, nEx, nEy, 0)

    # Boundary conditions: Fixed left edge
    add_boundary_condition(config, mesh, "west", DOF_U, 0)
    add_boundary_condition(config, mesh, "west", DOF_V, 0)

    # Apply tip load as distributed traction on right edge
    # Total force = traction * (edge_height * thickness) = t_y * (Ly * h)
    # Therefore: t_y = P / (Ly * h)
    t_y = -P / (Ly * h)  # Negative for downward direction
    load_func = lambda x, y: (0, t_y)
    add_load(config, mesh, "east", LOAD_TYPE_TRACTION, load_func)

    # Solve
    solver_data = create_solver_data(config, mesh)
    solve(config, solver_data, mesh)

    # Extract displacements
    u, v = unpack_solution(config, solver_data.r)

    #   Plot(config, mesh, solver_data)

    # Get displacement along the centerline (y = Ly/2)
    centerline_tolerance = Ly / (2 * nEy) + 1e-10
    centerline_mask = np.abs(mesh.nodes[:, 1] - Ly / 2) < centerline_tolerance
    nodeIDs_centerline = np.where(centerline_mask)[0]
    x_centerline = mesh.nodes[nodeIDs_centerline, 0]

    # Sort nodes by x-coordinate
    sorted_indices = np.argsort(x_centerline)
    nodeIDs_centerline_ordered = nodeIDs_centerline[sorted_indices]
    x_centerline_ordered = mesh.nodes[nodeIDs_centerline_ordered, 0]
    v_centerline_ordered = v[nodeIDs_centerline_ordered]

    return x_centerline_ordered, v_centerline_ordered, E, P, Ly, h


def analytical_beam_deflection(x, E, P, Lx, Ly, h):
    """Calculate analytical Euler-Bernoulli beam deflection for tip-loaded cantilever."""
    I = Ly**3 * h / 12  # Second moment of inertia for rectangular cross-section
    v_theory = -P / (6 * E * I) * (3 * Lx * x**2 - x**3)
    return v_theory


if __name__ == "__main__":
    print("=" * 70)
    print("BEAM COMPARISON EXAM TASK")
    print("=" * 70)

    # Problem parameters
    Lx = 1.0  # Length in x-direction
    Ly = 0.1  # Length in y-direction (beam height)

    # Test configurations
    mesh_configs = [
        (40, 4, "Mesh A: 40×4"),
        (10, 4, "Mesh B: 10×4"),
        (40, 1, "Mesh C: 40×1"),
    ]

    # Set up the plot with larger font sizes
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 18,
        'axes.titlesize': 20,
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'legend.fontsize': 15,
    })

    fig, ax = plt.subplots(figsize=(12, 8))

    # Get material properties from first run
    x_analytical = None
    v_analytical = None

    # Colors from matplotlib's default color cycle
    colors = ['C0', 'C1', 'C2']

    # Run simulations and plot
    for idx, (nEx, nEy, label) in enumerate(mesh_configs):
        print(f"\nRunning {label}...")
        x, v, E, P, Ly_sim, h = run_beam_test(nEx, nEy, Lx, Ly)

        # Calculate analytical solution on first iteration
        if x_analytical is None:
            x_analytical = np.linspace(0, Lx, 200)
            v_analytical = analytical_beam_deflection(x_analytical, E, P, Lx, Ly, h)

        # Plot FEM solution
        ax.plot(x,
                v,
                marker='none',
                markersize=6,
                linewidth=2.5,
                color=colors[idx],
                label=label,
                markevery=max(1,
                              len(x) // 20))

        # Print results
        print(f"  Number of elements: {nEx * nEy}")
        print(f"  Tip displacement: {v[-1]:.6e} m")

    # Plot analytical solution
    ax.plot(x_analytical, v_analytical, 'k--', linewidth=2.5, label='Euler-Bernoulli theory', alpha=0.7)

    # Formatting
    ax.set_xlabel('x', fontweight='bold', fontsize=22)
    ax.set_ylabel('v(x)', fontweight='bold', fontsize=22)
    #ax.set_title('Cantilever Beam Deflection Comparison', fontweight='bold', pad=20)
    ax.grid(True)
    ax.legend(fontsize=17, loc='lower left', framealpha=0.95)
    ax.tick_params(labelsize=15)

    plt.tight_layout()

    # Save figure with high DPI
    output_path = '/home/anders/projects/fem4146/exam-tasks/beam_comparison.png'
    plt.savefig(output_path, dpi=600, bbox_inches='tight')

    print("\n" + "=" * 70)
    print(f"Figure saved to: {output_path}")
    print("=" * 70)

    # Also show the plot
    plt.show()
