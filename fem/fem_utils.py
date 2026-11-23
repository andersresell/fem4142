import numpy as np
import signal
from enum import IntEnum
from dataclasses import dataclass

signal.signal(signal.SIGINT, signal.SIG_DFL)  #Makes Ctrl+C in the terminal close the plot window

SMALL_VAL = 1e-8

#====================================================================
# Named constants. An IntEnum is an integer with a name, so that we
# can write e.g. ElementType.Q16 instead of a number. The name can be
# printed with element_type.name (which gives "Q16").
#====================================================================


class ProblemType(IntEnum):
    PLANE_STRESS = 0
    PLATE = 1  #Mindlin plate


class ElementType(IntEnum):
    Q4 = 0
    Q4R = 1  #The R stands for reduced integration (fewer Gauss points)
    Q8 = 2
    Q8R = 3
    Q9 = 4
    Q9R = 5
    Q16 = 6
    Q4_USER = 7  #Used for creating tasks for students to implement an element


class Dof(IntEnum):
    """Degrees of freedom: U and V for plane stress problems, and W, THETAX and THETAY for plate problems"""
    U = 0
    V = 1
    W = 2
    THETAX = 3
    THETAY = 4


class LoadType(IntEnum):
    TRACTION = 0
    PRESSURE = 1
    BODY_FORCE = 2


class ElementEntity(IntEnum):
    """A part of a quad element: One of its four edges, or the whole face (see the figure in mesh.py)"""
    EDGE0 = 0
    EDGE1 = 1
    EDGE2 = 2
    EDGE3 = 3
    FACE = 4


class ElementCategory(IntEnum):
    LAGRANGIAN = 0  #The shape functions are products of 1D shape functions (Q4, Q9, Q16)
    SERENDIPITY = 1  #No internal nodes (Q8)


def get_num_dofs_from_problem_type(problem_type):
    if problem_type == ProblemType.PLANE_STRESS:
        return 2
    else:
        assert problem_type == ProblemType.PLATE
        return 3


def get_dof_id(dof, problem_type):
    """The position of the dof within a node, e.g. 1 for V in plane stress (u, v) and 0 for W in plates (w, thetax, thetay)"""
    if problem_type == ProblemType.PLANE_STRESS:
        dof_id = dof - Dof.U
        assert dof_id >= 0 and dof_id < 2
    else:
        assert problem_type == ProblemType.PLATE
        dof_id = dof - Dof.W
        assert dof_id >= 0 and dof_id < 3
    return dof_id


def has_hourglass_stiffness(element_type, problem_type):
    """True if the element has hourglass stabilization: An extra stiffness against the spurious zero energy modes
    from reduced integration (see calc_Ke_hourglass). For now, only Q4R in plane stress has it."""
    return element_type == ElementType.Q4R and problem_type == ProblemType.PLANE_STRESS


#====================================================================
# Element traits
#====================================================================

#The four edges of a quad element, given by their corner nodes (counter-clockwise)
QUAD_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]


@dataclass
class ElementTraits:
    """Describes an element type. One is created for each element type, in ELEMENT_TRAITS below.
    The nodes are numbered with the corners first, then the nodes on the edges, and finally the
    nodes inside the element."""
    category: ElementCategory  #Lagrangian or serendipity shape functions
    nGauss_1D: int  #Number of Gauss points in each direction, used for the stiffness matrix
    xi_eta: list  #Natural coordinates (xi, eta) of each node
    nNl: int  #Number of nodes
    nNl_1D: int  #Number of nodes along an edge
    il_to_ij: list  #Position (i, j) of each node in the grid of nodes along xi and eta
    corner_nodes: list
    internal_edge_nodes: list  #For each of the four edges: The nodes between the two corners
    internal_nodes: list  #The nodes inside the element
    entity_nodes: dict  #The nodes on each edge and on the face, e.g. entity_nodes[ElementEntity.EDGE0]

    def is_corner_node(self, il):
        return il in self.corner_nodes

    def is_internal_node(self, il):
        return il in self.internal_nodes

    def get_edge_id(self, il):
        """Returns the edge that the (non-corner) edge node il lies on"""
        for edge_id in range(4):
            if il in self.internal_edge_nodes[edge_id]:
                return edge_id
        assert False, f"Node {il} is not an edge node"

    def get_internal_edge_local_index(self, il, edge_id):
        assert il in self.internal_edge_nodes[edge_id]
        return self.internal_edge_nodes[edge_id].index(il)


def create_element_traits(category, nGauss_1D, xi_eta, il_to_ij) -> ElementTraits:
    """Creates the traits of an element type. The nodes must be ordered with the corners first, then the nodes
    on the edges, and finally the nodes inside the element. Each group is ordered counter-clockwise."""
    nNl = len(xi_eta)
    nNl_1D = il_to_ij[1][0] + 1  #Number of nodes along an edge (node 1 is the bottom right corner)
    n = nNl_1D - 1  #Number of node spacings along an edge

    #Sort the nodes that are not corners into the four edges (bottom, right, top, left) and the inside
    internal_edge_nodes = [[], [], [], []]
    internal_nodes = []
    for il in range(4, nNl):
        i, j = il_to_ij[il]
        if j == 0:
            internal_edge_nodes[0].append(il)
        elif i == n:
            internal_edge_nodes[1].append(il)
        elif j == n:
            internal_edge_nodes[2].append(il)
        elif i == 0:
            internal_edge_nodes[3].append(il)
        else:
            internal_nodes.append(il)

    entity_nodes = {ElementEntity.FACE: list(range(nNl))}
    for edge_id, (corner_a, corner_b) in enumerate(QUAD_EDGES):
        entity_nodes[ElementEntity(edge_id)] = [corner_a] + internal_edge_nodes[edge_id] + [corner_b]

    return ElementTraits(category, nGauss_1D, xi_eta, nNl, nNl_1D, il_to_ij, [0, 1, 2, 3], internal_edge_nodes,
                         internal_nodes, entity_nodes)


#====================================================================
# Natural coordinates (xi, eta) of the nodes of each element
#====================================================================
#fmt: off
XI_ETA_Q4 = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
XI_ETA_Q8 = XI_ETA_Q4 + [(0, -1), (1, 0), (0, 1), (-1, 0)]
XI_ETA_Q9 = XI_ETA_Q8 + [(0, 0)]
XI_ETA_Q16 = XI_ETA_Q4 + [(-1/3, -1), (1/3, -1), (1, -1/3), (1, 1/3), (1/3, 1), (-1/3, 1), (-1, 1/3), (-1, -1/3),
                          (-1/3, -1/3), (1/3, -1/3), (1/3, 1/3), (-1/3, 1/3)]

#====================================================================
# Position (i, j) of each node in the grid of nodes along xi and eta
#====================================================================
IL_TO_IJ_Q4 = [(0, 0), (1, 0), (1, 1), (0, 1)]
IL_TO_IJ_Q8 = [(0, 0), (2, 0), (2, 2), (0, 2), (1, 0), (2, 1), (1, 2), (0, 1)]
IL_TO_IJ_Q9 = IL_TO_IJ_Q8 + [(1, 1)]
IL_TO_IJ_Q16 = [(0, 0), (3, 0), (3, 3), (0, 3), (1, 0), (2, 0), (3, 1), (3, 2), (2, 3), (1, 3), (0, 2), (0, 1),
                (1, 1), (2, 1), (2, 2), (1, 2)]

#====================================================================
# The traits of all element types. They are created once, when the
# program starts. Use e.g. ELEMENT_TRAITS[ElementType.Q9].nNl
#====================================================================
ELEMENT_TRAITS = {
    #                                          shape functions              Gauss points in 1D  xi_eta      il_to_ij
    ElementType.Q4:      create_element_traits(ElementCategory.LAGRANGIAN,  2,                  XI_ETA_Q4,  IL_TO_IJ_Q4),
    ElementType.Q4R:     create_element_traits(ElementCategory.LAGRANGIAN,  1,                  XI_ETA_Q4,  IL_TO_IJ_Q4),
    ElementType.Q4_USER: create_element_traits(ElementCategory.LAGRANGIAN,  2,                  XI_ETA_Q4,  IL_TO_IJ_Q4),
    ElementType.Q8:      create_element_traits(ElementCategory.SERENDIPITY, 3,                  XI_ETA_Q8,  IL_TO_IJ_Q8),
    ElementType.Q8R:     create_element_traits(ElementCategory.SERENDIPITY, 2,                  XI_ETA_Q8,  IL_TO_IJ_Q8),
    ElementType.Q9:      create_element_traits(ElementCategory.LAGRANGIAN,  3,                  XI_ETA_Q9,  IL_TO_IJ_Q9),
    ElementType.Q9R:     create_element_traits(ElementCategory.LAGRANGIAN,  2,                  XI_ETA_Q9,  IL_TO_IJ_Q9),
    ElementType.Q16:     create_element_traits(ElementCategory.LAGRANGIAN,  4,                  XI_ETA_Q16, IL_TO_IJ_Q16),
}
#fmt: on
