import fem.shape_functions as shape_functions
from fem.utils import Config
from fem.mesh import Mesh
from fem.fem_utils import *


def recover_stress_plane_stress_consistent(e, config: Config, mesh: Mesh, nodes, u, v, xi_eta: list[tuple]):
    """Evaluates the stress sigma = D * B * r_e directly from the displacement field of element e at the
    sample points xi_eta (natural coordinates). This is the stress field that is consistent with the
    finite element formulation. It is in general discontinuous between elements.
    Returns an array of shape (number of sample points, 3) with the columns sigma_xx, sigma_yy, sigma_xy."""
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
