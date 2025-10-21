import src.shape_functions as shape_functions
from src.utils import Config
from src.mesh import Mesh
from src.fem_utils import *


def recover_stress_plane_stress_consistent(e, config: Config, mesh: Mesh, nodes, u, v, xi_eta: list[tuple]):
    assert config.problem_type == PROBLEM_TYPE_PLANE_STRESS
    assert e >= 0 and e < mesh.get_nE()
    elements = mesh.elements
    element_type = config.element_type
    nNl = element_type_to_nNl[element_type]
    E = config.E
    nu = config.nu
    n_sample_points = len(xi_eta)

    #fmt: off
    D = E / (1 - nu**2) * np.array([[1,     nu,     0],
                                    [nu,    1,      0],
                                    [0,     0,  (1 - nu) / 2]])
    #fmt: on
    arr_sigma = np.zeros((n_sample_points, 3))
    x_l = nodes[elements[e, :], 0]
    y_l = nodes[elements[e, :], 1]
    r_e = np.zeros(2 * nNl)
    r_e[0::2] = u[elements[e, :]]
    r_e[1::2] = v[elements[e, :]]

    assert len(x_l) == nNl and len(y_l) == nNl and len(r_e) == 2 * nNl
    for i in range(n_sample_points):
        xi, eta = xi_eta[i]
        dNdx, dNdy = shape_functions.calc_dNdx_dNdy(xi, eta, x_l, y_l, element_type)
        B = np.zeros((3, 2 * nNl))
        for k in range(nNl):
            #fmt: off
            B[:, 2 * k:2 * k + 2] = np.array([[dNdx[k],     0],
                                                [0,       dNdy[k]],
                                                [dNdy[k], dNdx[k]]])
            #fmt: on
        arr_sigma[i, :] = D @ B @ r_e
    return arr_sigma


#====================================================================
# These maps the element type to the element which corresponds to the
# gauss rule. Barlow used one order lower for fully integrated elements.
# For same order, then e.g Q8 -> Q9, Q9 -> Q9, Q16 -> Q16 etc.
#====================================================================
element_type_to_stress_recovery_extrapolation_element_barlow = {
    ELEMENT_TYPE_Q4R: ELEMENT_TYPE_Q4R,
    ELEMENT_TYPE_Q4: ELEMENT_TYPE_Q4R,
    ELEMENT_TYPE_Q4_USER: ELEMENT_TYPE_Q4R,
    ELEMENT_TYPE_Q8: ELEMENT_TYPE_Q4,
    ELEMENT_TYPE_Q8R: ELEMENT_TYPE_Q4,
    ELEMENT_TYPE_Q9: ELEMENT_TYPE_Q4,
    ELEMENT_TYPE_Q9R: ELEMENT_TYPE_Q4,
    ELEMENT_TYPE_Q16: ELEMENT_TYPE_Q9
}

element_type_to_stress_recovery_extrapolation_element_same_order = {
    ELEMENT_TYPE_Q4R: ELEMENT_TYPE_Q4R,
    ELEMENT_TYPE_Q4: ELEMENT_TYPE_Q4,
    ELEMENT_TYPE_Q4_USER: ELEMENT_TYPE_Q4,
    ELEMENT_TYPE_Q8: ELEMENT_TYPE_Q9,
    ELEMENT_TYPE_Q8R: ELEMENT_TYPE_Q9,
    ELEMENT_TYPE_Q9: ELEMENT_TYPE_Q9,
    ELEMENT_TYPE_Q9R: ELEMENT_TYPE_Q9R,
    ELEMENT_TYPE_Q16: ELEMENT_TYPE_Q16
}


def get_arr_xi(nGauss_1D):
    """Returns the natural coordinates for the Gauss quadrature rule for 1D integration."""
    if nGauss_1D == 4:
        arr_xi = np.array([
            -np.sqrt(3 / 7 + 2 / 7 * np.sqrt(6 / 5)), -np.sqrt(3 / 7 - 2 / 7 * np.sqrt(6 / 5)),
            np.sqrt(3 / 7 - 2 / 7 * np.sqrt(6 / 5)),
            np.sqrt(3 / 7 + 2 / 7 * np.sqrt(6 / 5))
        ])
    elif nGauss_1D == 3:
        arr_xi = np.array([-np.sqrt(3 / 5), 0, np.sqrt(3 / 5)])
    elif nGauss_1D == 2:
        arr_xi = np.array([-1 / np.sqrt(3), 1 / np.sqrt(3)])
    elif nGauss_1D == 1:
        arr_xi = np.array([0])
    else:
        assert False
    assert abs(np.sum(arr_xi)) < SMALL_VAL
    return arr_xi


#this is based on 1/xi_1D at the outermost gauss point
stress_recovery_extrapolation_element_to_scale = {
    ELEMENT_TYPE_Q4: np.sqrt(3),
    ELEMENT_TYPE_Q9: np.sqrt(5 / 3),
    ELEMENT_TYPE_Q16: 1.0 / np.sqrt(3 / 7 + 2 / 7 * np.sqrt(6 / 5))
}

# def calc_N_1D_recovery(xi, element_type_recovery, dir):
#     nNl_1d = element_type_to_nNl_1D[element_type_recovery]
#     xi_1D = shape_functions.get_arr_xi(nNl_1d)
#     il_to_ij = shape_functions.il_to_ij_loc_all[element_type_recovery]
#     r = np.zeros(nNl_1d)

#     for il in range(nNl_1d):
#         if dir == 0:
#             i = il_to_ij[il][0]
#         else:
#             assert dir == 1
#             i = il_to_ij[il][1]
#         xi_node = -1.0 + 2.0 * i / (nNl_1d - 1)
#         if np.isclose(xi_1D[il], 0.0):
#             scale = 1.0
#         else:
#             scale = abs(xi_node) / xi_1D[il]
#         r[i] = xi * scale
#     N = np.zeros(nNl_1d)
#     if nNl_1d == 2:
#         N[0] = 0.5 * (1 - r[0])
#         N[1] = 0.5 * (1 + r[1])
#     elif nNl_1d == 3:
#         N[0] = 0.5 * r[0] * (r[0] - 1)
#         N[1] = 1 - r[1]**2
#         N[2] = 0.5 * r[2] * (r[2] + 1)
#     elif nNl_1d == 4:
#         N[0] = -(9.0 / 16) * (r[0] + 1.0 / 3) * (r[0] - 1.0 / 3) * (r[0] - 1)
#         N[1] = (27.0 / 16) * (r[1] + 1) * (r[1] - 1.0 / 3) * (r[1] - 1)
#         N[2] = (-27.0 / 16) * (r[2] + 1) * (r[2] + 1.0 / 3) * (r[2] - 1)
#         N[3] = (9.0 / 16) * (r[3] + 1) * (r[3] + 1.0 / 3) * (r[3] - 1.0 / 3)
#     else:
#         assert False
#     return N

# def calc_N_recovery(xi, eta, element_type):
#     assert element_type_to_category(element_type) == ELEMENT_CATEGORY_LAGRANGIAN
#     nNl = element_type_to_nNl[element_type]
#     nNl_1d = element_type_to_nNl_1D[element_type]
#     N = np.zeros(nNl)
#     il_to_ij_loc = il_to_ij_loc_all[element_type]
#     N1D_xi = calc_N_1D_recovery(xi, element_type, 0)
#     N1D_eta = calc_N_1D_recovery(eta, element_type, 1)
#     for il in range(nNl):
#         i = il_to_ij_loc[il][0]
#         j = il_to_ij_loc[il][1]
#         N[il] = N1D_xi[i] * N1D_eta[j]
#     return N


def recover_stress_plane_stress_extrapolated(e, config: Config, mesh: Mesh, nodes, u, v, xi_eta: list[tuple]):
    assert config.problem_type == PROBLEM_TYPE_PLANE_STRESS
    assert e >= 0 and e < mesh.get_nE()
    method = config.stress_reconstruction_method
    elements = mesh.elements
    element_type = config.element_type
    nNl = element_type_to_nNl[element_type]
    E = config.E
    nu = config.nu
    n_sample_points = len(xi_eta)

    #fmt: off
    D = E / (1 - nu**2) * np.array([[1,     nu,     0],
                                    [nu,    1,      0],
                                    [0,     0,  (1 - nu) / 2]])
    #fmt: on
    arr_sigma = np.zeros((n_sample_points, 3))
    x_l = nodes[elements[e, :], 0]
    y_l = nodes[elements[e, :], 1]
    r_e = np.zeros(2 * nNl)
    r_e[0::2] = u[elements[e, :]]
    r_e[1::2] = v[elements[e, :]]

    if method == STRESS_RECONSTRUCTION_EXTRAPOLATED_BARLOW:
        element_type_eval = element_type_to_stress_recovery_extrapolation_element_barlow[element_type]
    else:
        assert method == STRESS_RECONSTRUCTION_EXTRAPOLATED_FULL
        element_type_eval = element_type_to_stress_recovery_extrapolation_element_same_order[element_type]

    nGauss_1D_eval = shape_functions.element_type_to_nGauss_1D[element_type_eval]

    sigma_gauss = np.zeros((nGauss_1D_eval**2, 3))
    il_to_ij_eval = il_to_ij_loc_all[element_type_eval]

    IX_sigma = lambda i_, j_: i_ * nGauss_1D_eval + j_
    #====================================================================
    # First evaluate the stresses at the gauss points
    #====================================================================
    for k in range(len(il_to_ij_eval)):
        i, j = il_to_ij_eval[k]
        if nGauss_1D_eval > 1:
            xi = -1.0 + 2.0 * i / (nGauss_1D_eval - 1)
            eta = -1.0 + 2.0 * j / (nGauss_1D_eval - 1)
        else:
            xi = 0.0
            eta = 0.0
        assert abs(xi) <= 1.0 and abs(eta) <= 1.0

        dNdx, dNdy = shape_functions.calc_dNdx_dNdy(xi, eta, x_l, y_l, element_type)  # Use original element type
        B = np.zeros((3, 2 * nNl))
        for k in range(nNl):
            #fmt: off
            B[:, 2 * k:2 * k + 2] =np.array([[dNdx[k],     0],
                                                [0,       dNdy[k]],
                                                [dNdy[k], dNdx[k]]])
            #fmt: on
        ID = IX_sigma(i, j)
        sigma_gauss[ID, :] = D @ B @ r_e
        if nGauss_1D_eval == 1:
            break

    if nGauss_1D_eval == 1:
        assert sigma_gauss.shape[0] == 1
        #return a constant value for all sample points
        return np.tile(sigma_gauss[0, :], (n_sample_points, 1))

    scale = stress_recovery_extrapolation_element_to_scale[element_type_eval]
    # scale = 1
    for n in range(n_sample_points):
        xi_sample, eta_sample = xi_eta[n]
        assert abs(xi_sample) <= 1.0 and abs(eta_sample) <= 1.0
        xi_scaled = xi_sample * scale
        eta_scaled = eta_sample * scale
        N_recovery = shape_functions.calc_N(xi_scaled, eta_scaled, element_type_eval)
        assert len(N_recovery) == sigma_gauss.shape[0]
        for k in range(len(il_to_ij_eval)):
            i, j = il_to_ij_eval[k]
            ID = IX_sigma(i, j)
            arr_sigma[n, :] += N_recovery[k] * sigma_gauss[ID, :]

    return arr_sigma
