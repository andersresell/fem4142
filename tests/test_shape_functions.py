"""Checks of the shape functions and Gauss quadrature rules."""
import pytest
from problems import ALL_ELEMENT_TYPES, element_name

import fem.shape_functions as shape_functions
from fem.fem_utils import *

# A few arbitrary points inside the reference element [-1, 1] x [-1, 1]
SAMPLE_POINTS = [(0.0, 0.0), (0.3, -0.7), (-0.9, 0.45), (0.77, 0.12), (-1.0, 1.0)]


@pytest.mark.parametrize("element_type", ALL_ELEMENT_TYPES, ids=element_name)
def test_shape_function_is_one_at_its_own_node_and_zero_at_the_others(element_type):
    xi_eta_nodes = ELEMENT_TRAITS[element_type].xi_eta
    for il, (xi, eta) in enumerate(xi_eta_nodes):
        N = shape_functions.calc_N(xi, eta, element_type)
        expected = np.zeros(len(xi_eta_nodes))
        expected[il] = 1.0
        assert np.allclose(N, expected, atol=1e-12), f"Node {il} at (xi, eta) = ({xi}, {eta})"


@pytest.mark.parametrize("element_type", ALL_ELEMENT_TYPES, ids=element_name)
def test_shape_functions_sum_to_one(element_type):
    """Partition of unity: required for the element to represent rigid body translations."""
    for xi, eta in SAMPLE_POINTS:
        N = shape_functions.calc_N(xi, eta, element_type)
        dNdxi, dNdeta = shape_functions.calc_dNdxi_dNdeta(xi, eta, element_type)
        assert np.isclose(np.sum(N), 1.0)
        assert np.isclose(np.sum(dNdxi), 0.0)
        assert np.isclose(np.sum(dNdeta), 0.0)


@pytest.mark.parametrize("element_type", ALL_ELEMENT_TYPES, ids=element_name)
def test_shape_function_derivatives_match_finite_differences(element_type):
    step = 1e-6
    for xi, eta in SAMPLE_POINTS[:4]:
        dNdxi, dNdeta = shape_functions.calc_dNdxi_dNdeta(xi, eta, element_type)
        dNdxi_fd = (shape_functions.calc_N(xi + step, eta, element_type) -
                    shape_functions.calc_N(xi - step, eta, element_type)) / (2 * step)
        dNdeta_fd = (shape_functions.calc_N(xi, eta + step, element_type) -
                     shape_functions.calc_N(xi, eta - step, element_type)) / (2 * step)
        assert np.allclose(dNdxi, dNdxi_fd, atol=1e-7)
        assert np.allclose(dNdeta, dNdeta_fd, atol=1e-7)


@pytest.mark.parametrize("nGauss_1D", [1, 2, 3, 4])
def test_gauss_rule_integrates_polynomials_exactly(nGauss_1D):
    """An n-point Gauss rule integrates polynomials up to degree 2n-1 exactly over [-1, 1]."""
    arr_xi = shape_functions.get_arr_xi(nGauss_1D)
    arr_w = shape_functions.get_arr_w(nGauss_1D)
    for degree in range(2 * nGauss_1D):
        exact = (1 - (-1)**(degree + 1)) / (degree + 1)
        assert np.isclose(np.sum(arr_w * arr_xi**degree), exact)
