"""
Convergence study with TIP LOAD for simple Euler-Bernoulli reference.
Tests Q4, Q4R, Q9 elements.
"""

import sys

sys.path.append('/home/anders/projects/fem4146')

from src.useful_imports import *
import matplotlib.pyplot as plt
import numpy as np

# Set up consistent plot style
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 18,
    'axes.titlesize': 20,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 15,
})


def run_beam_simulation(element_type, nEx, nEy, verbose=False):
    """Run cantilever beam simulation with tip load."""

    # Material and geometry
    E = 210e9  # Young's modulus in Pa
    nu = 0.3  # Poisson's ratio
    h = 1.0  # Plate thickness in m
    Lx = 10.0  # Length in x-direction
    Ly = 1.0  # Length in y-direction (beam height)
    P = 1000  # Tip load in N

    problem_type = PROBLEM_TYPE_PLANE_STRESS

    # Configuration
    config = create_config(E, nu, h, element_type, problem_type)
    config.hourglass_scaling = 0.1

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

    if verbose:
        elem_name = {0: 'Q4', 1: 'Q4R', 2: 'Q8', 4: 'Q9', 6: 'Q16'}.get(element_type, f'Type{element_type}')
        print(f"  {elem_name:4s} Mesh: {nEx:3d}×{nEy:3d}, DOFs: {len(solver_data.r):6d}", end='')

    # Plot(config, mesh, solver_data)  # Disabled for convergence study
    return u, v, mesh


def run_convergence_study():
    """Run convergence study for different element types."""

    print("=" * 70)
    print("CONVERGENCE STUDY: Cantilever Beam with Tip Load")
    print("=" * 70)

    # Problem parameters
    E = 210e9
    Lx = 10.0
    Ly = 1.0
    h = 1.0
    P = 1000

    # Analytical Euler-Bernoulli solution
    I = Ly**3 * h / 12  # Second moment of inertia
    tip_v_analytical = -P * Lx**3 / (3 * E * I)

    print(f"\nEuler-Bernoulli analytical solution:")
    print(f"  Tip displacement: {tip_v_analytical*1000:.6f} mm")
    print(f"  (v_tip = -PL³/(3EI))")

    # Element types to test
    element_types = {
        'Q4': ELEMENT_TYPE_Q4,
        'Q4R': ELEMENT_TYPE_Q4R,
        'Q9': ELEMENT_TYPE_Q9,
    }

    # Mesh refinement levels
    mesh_levels = [
        (2, 1),
        (4, 2),
        (8, 4),
        (16, 8),
        (32, 16),
        (64, 32),
    ]

    # Storage for results
    results = {elem_name: {'h': [], 'tip_v': [], 'error': []} for elem_name in element_types.keys()}

    # Run convergence study
    print("\n" + "-" * 70)
    print("Running convergence study...")
    print("-" * 70)

    for elem_name, elem_type in element_types.items():
        print(f"\n{elem_name}:")

        for nEx, nEy in mesh_levels:
            # Run simulation
            u, v, mesh = run_beam_simulation(elem_type, nEx, nEy, verbose=True)

            # Mesh size
            h = Lx / nEx

            # Find tip displacement (rightmost middle node)
            tip_node_idx = np.argmin(np.linalg.norm(mesh.nodes - np.array([Lx, Ly / 2]), axis=1))
            tip_v = v[tip_node_idx]

            # Compute relative error
            relative_error = abs((tip_v - tip_v_analytical) / tip_v_analytical)

            # Store results
            results[elem_name]['h'].append(h)
            results[elem_name]['tip_v'].append(tip_v)
            results[elem_name]['error'].append(relative_error)

            print(f", h={h:.3f}, error={relative_error:.4e}, tip_v={tip_v*1000:.5f} mm")

    # Create plots
    create_exam_plots(results, tip_v_analytical)

    return results, tip_v_analytical


def create_exam_plots(results, tip_v_analytical):
    """Create exam plot and answer key."""

    # Use mesh configurations for x-axis labels (nEx × nEy)
    mesh_configs = ['2×1', '4×2', '8×4', '16×8', '32×16', '64×32']
    x_positions = np.arange(len(mesh_configs))

    # EXAM VERSION (unlabeled)
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    styles = {
        'Q4': {
            'color': '#1f77b4',
            'marker': 'o',
            'markersize': 12,
            'linewidth': 3.5,
            'label': 'Element A'
        },
        'Q4R': {
            'color': '#ff7f0e',
            'marker': 's',
            'markersize': 12,
            'linewidth': 3.5,
            'label': 'Element B'
        },
        'Q9': {
            'color': '#2ca02c',
            'marker': 'd',
            'markersize': 13,
            'linewidth': 3.5,
            'label': 'Element C'
        },
    }

    for elem_name, data in results.items():
        tip_v = np.array(data['tip_v'])  # Keep in meters
        style = styles[elem_name]

        ax.plot(x_positions,
                tip_v,
                marker=style['marker'],
                color=style['color'],
                label=style['label'],
                linewidth=style['linewidth'],
                markersize=style['markersize'],
                markeredgewidth=1.5,
                markeredgecolor='white')

    # Add analytical reference line
    ax.axhline(tip_v_analytical,
               color='black',
               linestyle='--',
               linewidth=2.5,
               alpha=0.7,
               label='Euler-Bernoulli theory')

    ax.set_xlabel('Mesh Refinement', fontweight='bold')
    ax.set_ylabel('Tip displacement', fontweight='bold')
    # Title removed
    ax.set_xticks(x_positions)
    ax.set_xticklabels(mesh_configs, rotation=0)
    ax.grid(True)
    ax.legend(fontsize=17, framealpha=0.95, loc='lower right')
    ax.tick_params(labelsize=15)

    plt.tight_layout()
    plt.savefig('/home/anders/projects/fem4146/exam-tasks/exam_plot_FINAL_UNLABELED.png', dpi=600, bbox_inches='tight')
    print("\n" + "=" * 70)
    print("Unlabeled exam plot saved to: exam-tasks/exam_plot_FINAL_UNLABELED.png")

    # ANSWER KEY VERSION
    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 8))

    styles_answer = {
        'Q4': {
            'color': '#1f77b4',
            'marker': 'o',
            'markersize': 12,
            'linewidth': 3.5
        },
        'Q4R': {
            'color': '#ff7f0e',
            'marker': 's',
            'markersize': 12,
            'linewidth': 3.5
        },
        'Q9': {
            'color': '#2ca02c',
            'marker': 'd',
            'markersize': 13,
            'linewidth': 3.5
        },
    }

    for elem_name, data in results.items():
        tip_v = np.array(data['tip_v'])  # Keep in meters
        style = styles_answer[elem_name]

        # Add error at coarsest mesh to label
        error_coarse = abs((data['tip_v'][0] - tip_v_analytical) / tip_v_analytical) * 100
        label = f'{elem_name} (initial error: {error_coarse:.1f}%)'

        ax2.plot(x_positions,
                 tip_v,
                 marker=style['marker'],
                 color=style['color'],
                 label=label,
                 linewidth=style['linewidth'],
                 markersize=style['markersize'],
                 markeredgewidth=1.5,
                 markeredgecolor='white')

    ax2.axhline(tip_v_analytical,
                color='black',
                linestyle='--',
                linewidth=2.5,
                alpha=0.7,
                label='Euler-Bernoulli theory')

    ax2.set_xlabel('Mesh Refinement', fontweight='bold')
    ax2.set_ylabel('Tip displacement', fontweight='bold')
    # Title removed
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(mesh_configs, rotation=0)
    ax2.grid(True, alpha=0.3, linestyle=':', axis='y')
    ax2.legend(fontsize=17, framealpha=0.95, loc='lower right')
    ax2.tick_params(labelsize=15)

    # Simpler annotations - just text, no arrows
    # Add key observations as text box
    textstr = ('Key Observations:\n'
               '• Element A: Severe shear locking (70% error)\n'
               '• Element B: Reduced integration helps\n'
               '• Element C: Fast, smooth convergence')
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='black', linewidth=1.5)
    ax2.text(0.02,
             0.98,
             textstr,
             transform=ax2.transAxes,
             fontsize=14,
             verticalalignment='top',
             horizontalalignment='left',
             bbox=props)

    plt.tight_layout()
    plt.savefig('/home/anders/projects/fem4146/exam-tasks/exam_plot_FINAL_ANSWER.png', dpi=600, bbox_inches='tight')
    print("Answer key plot saved to: exam-tasks/exam_plot_FINAL_ANSWER.png")
    print("=" * 70)

    plt.show()

    print("\n" + "=" * 70)
    print("ANSWER KEY")
    print("=" * 70)
    print("Element A = Q4   (severe shear locking, converges from below)")
    print("Element B = Q4R  (reduced integration, much better)")
    print("Element C = Q9   (quadratic, fast smooth convergence)")
    print("=" * 70)


if __name__ == "__main__":
    results, tip_v_analytical = run_convergence_study()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Analytical (Euler-Bernoulli): {tip_v_analytical*1000:.6f} mm")
    print(f"\nFinal converged values (h=0.156 m):")
    for elem_name, data in results.items():
        final_v = data['tip_v'][-1] * 1000
        error = data['error'][-1] * 100
        print(f"  {elem_name:4s}: {final_v:.5f} mm (error: {error:.3f}%)")
    print("=" * 70)
