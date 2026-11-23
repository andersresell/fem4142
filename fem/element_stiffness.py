import fem.shape_functions as shape_functions
from fem.utils import Config
from fem.mesh import Mesh
from fem.fem_utils import *


def calc_Ke_plane_stress(config: Config, mesh: Mesh, e):
    assert config.problem_type == ProblemType.PLANE_STRESS
    assert e >= 0 and e < mesh.get_nE()
    element_type = config.element_type
    E = config.E
    nu = config.nu
    thickness = config.h
    elements = mesh.elements
    x_l = mesh.nodes[elements[e, :], 0]
    y_l = mesh.nodes[elements[e, :], 1]
    nNl = ELEMENT_TRAITS[element_type].nNl
    assert len(x_l) == nNl and len(x_l) == nNl

    #fmt: off
    D = E / (1 - nu**2) * np.array([[1,     nu,     0],
                                    [nu,    1,      0],
                                    [0,     0,  (1 - nu) / 2]])
    #fmt: on

    nGauss = ELEMENT_TRAITS[element_type].nGauss_1D
    arr_xi = shape_functions.get_arr_xi(nGauss)
    arr_w = shape_functions.get_arr_w(nGauss)

    Ke = np.zeros((2 * nNl, 2 * nNl))

    for i in range(nGauss):
        for j in range(nGauss):
            xi = arr_xi[i]
            eta = arr_xi[j]
            weight = arr_w[i] * arr_w[j]
            N = shape_functions.calc_N(xi, eta, element_type)
            dNdx, dNdy = shape_functions.calc_dNdx_dNdy(xi, eta, x_l, y_l, element_type)
            J = shape_functions.calc_J(xi, eta, x_l, y_l, element_type)
            detJ = np.linalg.det(J)
            assert len(N) == nNl and len(dNdx) == nNl and len(dNdy) == nNl
            B = np.zeros((3, 2 * nNl))
            for k in range(nNl):
                #fmt: off
                B[:, 2 * k:2 * k + 2] = np.array([[dNdx[k],     0],
                                                  [0,       dNdy[k]],
                                                  [dNdy[k], dNdx[k]]])
                #fmt: on

            Ke += B.T @ D @ B * detJ * weight * thickness

    return Ke


def calc_Ke_hourglass_plane_stress_Q4R(config: Config, mesh: Mesh, e):
    assert config.element_type == ElementType.Q4R
    #====================================================================
    # Add hourglass stabilization term: See Page 522, eq. (8.7.11) in
    # the book Nonlinear Finite Elements for Continua and Structures, by
    # Belytschko, Liu and Moran
    #====================================================================

    #For reduced integration of Q4R, we only have one Gauss point at the centroid
    xi = 0
    eta = 0
    element_type = config.element_type
    E = config.E
    nu = config.nu
    thickness = config.h
    elements = mesh.elements
    x_l = mesh.nodes[elements[e, :], 0]
    y_l = mesh.nodes[elements[e, :], 1]
    nNl = ELEMENT_TRAITS[element_type].nNl
    assert len(x_l) == nNl and len(x_l) == nNl
    dNdx, dNdy = shape_functions.calc_dNdx_dNdy(xi, eta, x_l, y_l, element_type)
    J = shape_functions.calc_J(xi, eta, x_l, y_l, element_type)
    detJ = np.linalg.det(J)
    #Shape function derivatives at the element center
    b_x, b_y = dNdx, dNdy
    A = 4 * detJ  #Element area
    h = np.array([1, -1, 1, -1]).transpose()
    gamma = 1 / 4 * (h - (h.dot(x_l)) * b_x - (h.dot(y_l)) * b_y)
    #====================================================================
    # Need to do this since the formulation in the book orders the dofs as
    # [u1, u2, u3, u4, v1, v2, v3, v4] whereas we use
    # [u1, v1, u2, v2, u3, v3, u4, v4]
    #====================================================================
    gamma_x = np.array([gamma[0], 0, gamma[1], 0, gamma[2], 0, gamma[3], 0])
    gamma_y = np.array([0, gamma[0], 0, gamma[1], 0, gamma[2], 0, gamma[3]])

    mu = E / (2 * (1 + nu))
    #Need to check the scaling C_Q, as it's based on dynamic eigenvalue analysis
    C_Q = 0.5 * config.hourglass_scaling * mu * (b_x.dot(b_x) + b_y.dot(b_y))  #removed A

    #fmt: off
    K_hg = C_Q * thickness* A * (np.outer(gamma_x, gamma_x) + np.outer(gamma_y, gamma_y))
    #fmt: on

    return K_hg


def calc_Ke_mindlin_plate(config: Config, mesh: Mesh, e):
    assert config.problem_type == ProblemType.PLATE
    assert e >= 0 and e < mesh.get_nE()
    element_type = config.element_type
    E = config.E
    nu = config.nu
    h = config.h
    elements = mesh.elements
    x_l = mesh.nodes[elements[e, :], 0]
    y_l = mesh.nodes[elements[e, :], 1]
    nNl = ELEMENT_TRAITS[element_type].nNl
    assert len(x_l) == nNl and len(x_l) == nNl

    #fmt: off
    D = E * h**3 / (12 * (1 - nu**2)) * np.array([[1,   nu, 0],
                                                  [nu,  1,  0],
                                                  [0,   0,  (1 - nu) / 2]])
    #fmt: on
    Cs = E / (2 * (1 + nu)) * np.identity(2)

    hs = 5 / 6 * h

    nGauss = ELEMENT_TRAITS[element_type].nGauss_1D
    arr_xi = shape_functions.get_arr_xi(nGauss)
    arr_w = shape_functions.get_arr_w(nGauss)

    Ke = np.zeros((3 * nNl, 3 * nNl))

    for i in range(nGauss):
        for j in range(nGauss):
            xi = arr_xi[i]
            eta = arr_xi[j]
            weight = arr_w[i] * arr_w[j]
            N = shape_functions.calc_N(xi, eta, element_type)
            dNdx, dNdy = shape_functions.calc_dNdx_dNdy(xi, eta, x_l, y_l, element_type)
            J = shape_functions.calc_J(xi, eta, x_l, y_l, element_type)
            detJ = np.linalg.det(J)
            assert len(N) == nNl and len(dNdx) == nNl and len(dNdy) == nNl
            Bb = np.zeros((3, 3 * nNl))
            Bs = np.zeros((2, 3 * nNl))
            for k in range(nNl):
                #fmt: off
                Bb[:, 3 * k:3 * k + 3] = np.array([[0,  0,       -dNdx[k]],
                                                   [0,  dNdy[k],  0      ],
                                                   [0,  dNdx[k], -dNdy[k]]])
                Bs[:, 3 * k:3 * k + 3] = np.array([[dNdy[k], -N[k], 0    ],
                                                   [dNdx[k],  0,    N[k] ]])
                #fmt: on
            Ke += (Bb.T @ D @ Bb + hs * Bs.T @ Cs @ Bs) * detJ * weight

    return Ke
