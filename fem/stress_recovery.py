from scipy.sparse import lil_matrix
from scipy.sparse.linalg import splu
import fem.shape_functions as shape_functions
from fem.element_stiffness import calc_D_plane_stress, calc_D_plate
from fem.utils import Config
from fem.mesh import Mesh
from fem.fem_utils import *

#====================================================================
# Stress recovery (post-processing). There are two methods:
#   Consistent: sigma = D * B * r_e, calculated directly in each element. This is the stress that the
#               finite element solution gives. It is discontinuous between the elements.
#   L2:         A continuous stress field, interpolated from nodal values with the shape functions,
#               which is the best fit (in the least squares sense) to the consistent stress.
# For plates, the "stresses" are the stress resultants: The bending moments M and the shear forces V
# (per unit length).
#====================================================================

#====================================================================
# Shape functions at many points at once. This is much faster than
# calculating them point by point with shape_functions.py, which
# matters here (and in the plots), since we evaluate the stress at
# many points. The results are arrays with one row per point and one
# column per node.
#====================================================================


def calc_N_at_points(xi_eta, element_type):
    """Returns N, dN/dxi and dN/deta at the points xi_eta = [(xi_0, eta_0), (xi_1, eta_1), ...].
    They only depend on the element type, so they can be calculated once and used for all elements."""
    N = np.array([shape_functions.calc_N(xi, eta, element_type) for xi, eta in xi_eta])
    dN = [shape_functions.calc_dNdxi_dNdeta(xi, eta, element_type) for xi, eta in xi_eta]
    dNdxi = np.array([dNdxi_point for dNdxi_point, dNdeta_point in dN])
    dNdeta = np.array([dNdeta_point for dNdxi_point, dNdeta_point in dN])
    return N, dNdxi, dNdeta


def calc_dNdx_dNdy_at_points(dNdxi, dNdeta, x_l, y_l):
    """Returns dN/dx, dN/dy and det(J) at many points in an element with node coordinates x_l, y_l."""
    #The Jacobian J = [[dx/dxi, dy/dxi], [dx/deta, dy/deta]] at each point
    dxdxi, dydxi = dNdxi @ x_l, dNdxi @ y_l
    dxdeta, dydeta = dNdeta @ x_l, dNdeta @ y_l
    detJ = dxdxi * dydeta - dydxi * dxdeta
    assert np.all(detJ > SMALL_VAL)
    #[dN/dx, dN/dy] = inverse(J) * [dN/dxi, dN/deta], written out for a 2x2 matrix.
    #([:, None] makes the arrays with one value per point work together with arrays with one row per point)
    dNdx = (dydeta[:, None] * dNdxi - dydxi[:, None] * dNdeta) / detJ[:, None]
    dNdy = (-dxdeta[:, None] * dNdxi + dxdxi[:, None] * dNdeta) / detJ[:, None]
    return dNdx, dNdy, detJ


#====================================================================
# Stress recovery
#====================================================================


def get_stress_names(problem_type):
    """The stress components, in the order they are returned by the functions below"""
    if problem_type == ProblemType.PLANE_STRESS:
        return ["sigma_xx", "sigma_yy", "sigma_xy"]
    else:
        assert problem_type == ProblemType.PLATE
        return ["M_xx", "M_yy", "M_xy", "V_x", "V_y"]


def calc_element_stress(config: Config, mesh: Mesh, r, e, N, dNdxi, dNdeta):
    """Calculates the consistent stress in element e at several points at once. N, dNdxi and dNdeta are the shape
    functions at the points (see calc_N_at_points). r is the solution vector (solver_data.r).
    Returns an array with one row per point and one column per stress component (see get_stress_names)."""
    NUM_DOFS = get_num_dofs_from_problem_type(config.problem_type)
    nodeIDs = mesh.elements[e, :]
    x_l = mesh.nodes[nodeIDs, 0]
    y_l = mesh.nodes[nodeIDs, 1]
    r_e = r.reshape(-1, NUM_DOFS)[nodeIDs]  #One row per element node: (u, v) or (w, thetax, thetay)
    dNdx, dNdy, detJ = calc_dNdx_dNdy_at_points(dNdxi, dNdeta, x_l, y_l)

    if config.problem_type == ProblemType.PLANE_STRESS:
        u_e, v_e = r_e[:, 0], r_e[:, 1]
        eps_xx = dNdx @ u_e  #du/dx
        eps_yy = dNdy @ v_e  #dv/dy
        gamma_xy = dNdy @ u_e + dNdx @ v_e  #du/dy + dv/dx
        strain = np.column_stack([eps_xx, eps_yy, gamma_xy])
        D = calc_D_plane_stress(config)
        return strain @ D.T  #sigma = D * strain, for each point (row)
    else:
        assert config.problem_type == ProblemType.PLATE
        w_e, thetax_e, thetay_e = r_e[:, 0], r_e[:, 1], r_e[:, 2]
        #Curvatures and shear strains, defined in the same way as Bb and Bs in calc_Ke_mindlin_plate
        kappa = np.column_stack([-dNdx @ thetay_e, dNdy @ thetax_e, dNdx @ thetax_e - dNdy @ thetay_e])
        gamma = np.column_stack([dNdx @ w_e + N @ thetay_e, dNdy @ w_e - N @ thetax_e])  #gamma_xz, gamma_yz
        D_b, D_s = calc_D_plate(config)
        return np.column_stack([kappa @ D_b.T, gamma @ D_s.T])  #M = D_b * kappa and V = D_s * gamma


def calc_stress_consistent(config: Config, mesh: Mesh, r, e, xi_eta):
    """The consistent stress in element e at the points xi_eta = [(xi_0, eta_0), (xi_1, eta_1), ...].
    Returns an array with one row per point and one column per stress component (see get_stress_names)."""
    N, dNdxi, dNdeta = calc_N_at_points(xi_eta, config.element_type)
    return calc_element_stress(config, mesh, r, e, N, dNdxi, dNdeta)


def calc_nodal_stress_L2(config: Config, mesh: Mesh, r):
    """L2 projection: Finds the nodal stresses sigma_nodes, so that the continuous field N * sigma_nodes is as close
    as possible to the consistent stress sigma (it minimizes the integral of (N * sigma_nodes - sigma)^2).
    This gives the linear system M * sigma_nodes = (integral of N^T * sigma), where M = (integral of N^T * N) is the
    consistent mass matrix. Each stress component is found in the same way.
    Returns an array with one row per node and one column per stress component (see get_stress_names)."""
    element_type = config.element_type
    nNl = ELEMENT_TRAITS[element_type].nNl
    nN = mesh.get_nN()
    n_components = len(get_stress_names(config.problem_type))

    #Gauss points for full integration (as many points as nodes along an edge, in each direction)
    nGauss_1D = ELEMENT_TRAITS[element_type].nNl_1D
    arr_xi = shape_functions.get_arr_xi(nGauss_1D)
    arr_w = shape_functions.get_arr_w(nGauss_1D)
    xi_eta = [(xi, eta) for xi in arr_xi for eta in arr_xi]
    weights = np.array([w_i * w_j for w_i in arr_w for w_j in arr_w])
    N, dNdxi, dNdeta = calc_N_at_points(xi_eta, element_type)

    M = lil_matrix((nN, nN))
    rhs = [np.zeros(nN) for k in range(n_components)]  #One right-hand side vector per stress component
    for e in range(mesh.get_nE()):
        nodeIDs = mesh.elements[e, :]
        x_l = mesh.nodes[nodeIDs, 0]
        y_l = mesh.nodes[nodeIDs, 1]
        dNdx, dNdy, detJ = calc_dNdx_dNdy_at_points(dNdxi, dNdeta, x_l, y_l)
        dA = detJ * weights  #The area that each Gauss point represents
        sigma = calc_element_stress(config, mesh, r, e, N, dNdxi, dNdeta)  #At the Gauss points

        M_e = N.T @ (dA[:, None] * N)  #Integral of N^T * N
        for il in range(nNl):
            for jl in range(nNl):
                M[nodeIDs[il], nodeIDs[jl]] += M_e[il, jl]
        for k in range(n_components):
            rhs_e = N.T @ (dA * sigma[:, k])  #Integral of N^T * sigma_k
            for il in range(nNl):
                rhs[k][nodeIDs[il]] += rhs_e[il]

    #Each stress component has its own linear system with the same matrix M. So we factorize M once
    #(the expensive part), and use the factorization to solve the system for each component.
    M_factorized = splu(M.tocsc())
    nodal_stress = np.zeros((nN, n_components))
    for k in range(n_components):
        nodal_stress[:, k] = M_factorized.solve(rhs[k])
    return nodal_stress


def calc_von_mises(sigma):
    """Von Mises stress for plane stress. sigma has the columns sigma_xx, sigma_yy, sigma_xy (one row per point)"""
    sxx, syy, sxy = sigma[:, 0], sigma[:, 1], sigma[:, 2]
    return np.sqrt(sxx**2 - sxx * syy + syy**2 + 3 * sxy**2)
