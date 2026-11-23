"""Checks of the element traits in fem_utils.py."""
import pytest

from fem.fem_utils import ELEMENT_TRAITS, ElementEntity
from fem.useful_imports import *

# Written by hand. These are the tables that were in the code before they were calculated from il_to_ij.
#fmt: off
EXPECTED_TRAITS = {
    ElementType.Q4: dict(nNl_1D=2, internal_edge_nodes=[[], [], [], []], internal_nodes=[],
                         edges=[[0, 1], [1, 2], [2, 3], [3, 0]]),
    ElementType.Q8: dict(nNl_1D=3, internal_edge_nodes=[[4], [5], [6], [7]], internal_nodes=[],
                         edges=[[0, 4, 1], [1, 5, 2], [2, 6, 3], [3, 7, 0]]),
    ElementType.Q9: dict(nNl_1D=3, internal_edge_nodes=[[4], [5], [6], [7]], internal_nodes=[8],
                         edges=[[0, 4, 1], [1, 5, 2], [2, 6, 3], [3, 7, 0]]),
    ElementType.Q16: dict(nNl_1D=4, internal_edge_nodes=[[4, 5], [6, 7], [8, 9], [10, 11]], internal_nodes=[12, 13, 14, 15],
                          edges=[[0, 4, 5, 1], [1, 6, 7, 2], [2, 8, 9, 3], [3, 10, 11, 0]]),
}
#fmt: on


@pytest.mark.parametrize("element_type", EXPECTED_TRAITS.keys(), ids=lambda element_type: element_type.name)
def test_calculated_traits_match_the_tables_written_by_hand(element_type):
    traits = ELEMENT_TRAITS[element_type]
    expected = EXPECTED_TRAITS[element_type]
    assert traits.nNl == len(traits.xi_eta) == len(traits.il_to_ij)
    assert traits.nNl_1D == expected["nNl_1D"]
    assert traits.internal_edge_nodes == expected["internal_edge_nodes"]
    assert traits.internal_nodes == expected["internal_nodes"]
    for edge_id in range(4):
        assert traits.entity_nodes[ElementEntity(edge_id)] == expected["edges"][edge_id]
    assert traits.entity_nodes[ElementEntity.FACE] == list(range(traits.nNl))


@pytest.mark.parametrize("element_type", ELEMENT_TRAITS.keys(), ids=lambda element_type: element_type.name)
def test_xi_eta_and_il_to_ij_describe_the_same_node_positions(element_type):
    traits = ELEMENT_TRAITS[element_type]
    n = traits.nNl_1D - 1
    for (xi, eta), (i, j) in zip(traits.xi_eta, traits.il_to_ij):
        assert xi == pytest.approx(-1 + 2 * i / n) and eta == pytest.approx(-1 + 2 * j / n)


@pytest.mark.parametrize("full, reduced", [(ElementType.Q4, ElementType.Q4R), (ElementType.Q8, ElementType.Q8R),
                                           (ElementType.Q9, ElementType.Q9R)],
                         ids=["Q4R", "Q8R", "Q9R"])
def test_reduced_integration_uses_one_gauss_point_less_in_each_direction(full, reduced):
    assert ELEMENT_TRAITS[reduced].nGauss_1D == ELEMENT_TRAITS[full].nGauss_1D - 1
    assert ELEMENT_TRAITS[reduced].xi_eta == ELEMENT_TRAITS[full].xi_eta


def test_every_element_type_has_traits():
    assert set(ELEMENT_TRAITS.keys()) == set(ElementType)
