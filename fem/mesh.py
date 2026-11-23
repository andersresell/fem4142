from dataclasses import dataclass
import numpy as np
from fem.utils import *
from fem.fem_utils import *
import fem.shape_functions as shape_functions


@dataclass
class ElementSet:
    element_entity: ElementEntity  #The part of the elements that the set contains: one of the edges, or the face
    elementIDs: np.ndarray  #The elements in the set


@dataclass
class Mesh:
    nodes: np.ndarray  #Node coordinates, one row (x, y) per node
    elements: np.ndarray  #The node numbers of each element, one row per element
    node_sets: dict[str, np.ndarray]  #Named sets of node numbers, e.g. "west"
    element_sets: dict[str, ElementSet]  #Named sets of elements, e.g. "west"

    def get_nN(self):
        """Get number of nodes"""
        return self.nodes.shape[0]

    def get_nE(self):
        """Get number of elements"""
        return self.elements.shape[0]


def check_if_internal_edge_is_created(existing_edges_Q4, I_corner_a, I_corner_b):
    exists = False
    reversed = None
    if (I_corner_a, I_corner_b) in existing_edges_Q4:
        reversed = False
        exists = True
    elif (I_corner_b, I_corner_a) in existing_edges_Q4:
        reversed = True
        exists = True
    return exists, reversed


def create_higher_order_quad_mesh_from_Q4_mesh(mesh_Q4: Mesh, config: Config):
    element_traits = ELEMENT_TRAITS[config.element_type]
    nNl = element_traits.nNl

    elements_Q4 = mesh_Q4.elements
    nodes_Q4 = mesh_Q4.nodes
    assert elements_Q4.shape[1] == 4
    nE = elements_Q4.shape[0]

    nodes = nodes_Q4.copy()
    elements = np.full((nE, nNl), fill_value=-1, dtype=int)
    existing_edges_Q4 = {}
    new_node_counter = nodes_Q4.shape[0]
    for e in range(nE):

        #Retrieve the Q4 corner nodes
        x_Q4 = nodes_Q4[elements_Q4[e, :], 0]
        y_Q4 = nodes_Q4[elements_Q4[e, :], 1]

        for il in range(nNl):

            add_new_node = False
            I = -1
            if element_traits.is_corner_node(il):
                #If the node is a corner node, retrieve the Q4 corner node number
                I = elements_Q4[e, element_traits.corner_nodes.index(il)]
            elif element_traits.is_internal_node(il):
                #If the node is an internal node, we always add a new node
                add_new_node = True
            else:
                #Now we know that the node is an internal edge node
                edge_id = element_traits.get_edge_id(il)

                I_corner_a = elements_Q4[e, QUAD_EDGES[edge_id][0]]
                I_corner_b = elements_Q4[e, QUAD_EDGES[edge_id][1]]

                exists, reversed = check_if_internal_edge_is_created(existing_edges_Q4, I_corner_a, I_corner_b)
                if exists:
                    if reversed:
                        I_corner_tuple = (I_corner_b, I_corner_a)
                    else:
                        I_corner_tuple = (I_corner_a, I_corner_b)
                    #Internal edge node already exist
                    num_internal_edge_nodes = len(element_traits.internal_edge_nodes[edge_id])
                    assert num_internal_edge_nodes == len(existing_edges_Q4[I_corner_tuple])
                    local_index = element_traits.get_internal_edge_local_index(il, edge_id)
                    if reversed:
                        local_index = num_internal_edge_nodes - 1 - local_index
                    I = existing_edges_Q4[I_corner_tuple][local_index]
                else:
                    #Internal edge node doesn't exist, so we must add a new
                    add_new_node = True

            if add_new_node:
                I = new_node_counter
                new_node_counter += 1

                xi, eta = element_traits.xi_eta[il]
                N = shape_functions.calc_N(xi, eta, ElementType.Q4)
                #Calculate the new node coordinates
                x, y = N @ x_Q4, N @ y_Q4
                nodes = np.append(nodes, [[x, y]], axis=0)

            assert I not in elements[e, :]
            elements[e, il] = I

        #Cache all edges that wasn't previously created for this element
        for edge_id in range(4):
            I_corner_a = elements[e, QUAD_EDGES[edge_id][0]]
            I_corner_b = elements[e, QUAD_EDGES[edge_id][1]]
            if (I_corner_a, I_corner_b) not in existing_edges_Q4:
                existing_edges_Q4[(I_corner_a, I_corner_b)] = []
                for il in element_traits.internal_edge_nodes[edge_id]:
                    existing_edges_Q4[(I_corner_a, I_corner_b)].append(elements[e, il])

    #====================================================================
    # Create node sets from the previously created element sets
    #====================================================================

    node_sets = mesh_Q4.node_sets.copy()
    for set_name, elem_set in mesh_Q4.element_sets.items():
        nodeIDs = set()  #use a set to avoid duplicate nodeIDs
        elementIDs = elem_set.elementIDs
        nE_set = len(elementIDs)
        element_entity = elem_set.element_entity
        for e_l in range(nE_set):
            e = elementIDs[e_l]
            nodeIDs_l = element_traits.entity_nodes[element_entity]
            for il in nodeIDs_l:
                I = elements[e, il]
                nodeIDs.add(I)
        node_sets[set_name] = np.array(sorted(nodeIDs), dtype=int)

    return Mesh(nodes, elements, node_sets, mesh_Q4.element_sets.copy())


def create_structured_Q4_mesh(Lx, Ly, nEx, nEy) -> Mesh:

    #====================================================================
    # Create nodes and elements
    #====================================================================
    nE = nEx * nEy
    nNl = 4
    nNx = nEx + 1
    nNy = nEy + 1
    nN = nNx * nNy

    dx = Lx / nEx
    dy = Ly / nEy

    elements = np.zeros((nE, nNl), dtype=int)
    nodes = np.zeros((nN, 2))

    def IX_node(i, j):
        #maps 2D index to node number
        assert i >= 0 and j >= 0
        assert i < nNx and j < nNy
        I = j * nNx + i
        return I

    def IX_element(ex, ey):
        #maps 2D index to node number
        assert ex >= 0 and ey >= 0
        assert ex < nEx and ey < nEy
        e = ey * nEx + ex
        return e

    for j in range(nNy):
        for i in range(nNx):
            x = i * dx
            y = j * dy
            I = IX_node(i, j)
            nodes[I, :] = np.array([x, y])

    il_to_ij_loc = ELEMENT_TRAITS[ElementType.Q4].il_to_ij
    nNl_1D = 2
    for ey in range(nEy):
        for ex in range(nEx):
            e = IX_element(ex, ey)
            for il in range(nNl):
                i = (nNl_1D - 1) * ex + il_to_ij_loc[il][0]
                j = (nNl_1D - 1) * ey + il_to_ij_loc[il][1]
                elements[e, il] = IX_node(i, j)
    """Create predefined element sets
        Domain is shown below:
   
                          north
 north_west .______________________________.north_east
            |                              |
            |                              |
            |                              |
            |                              |
    west    |            domain            |east
            |                              | 
            |                              |
            |                              |
 south_west .______________________________.south_east
                          south 
                          
    Each quad element is marked with local element entities as follows

        ______EDGE2_____
        |               |
        |               |
   EDGE3|      FACE     |EDGE1
        |               |
        |               |
        |_______________|
              EDGE0
    Each element set is marked with a name and contains the element
    indices, and the element entity (edge or face) that the set refers to
    """
    set_name_to_entity = {
        "domain": ElementEntity.FACE,
        "west": ElementEntity.EDGE3,
        "east": ElementEntity.EDGE1,
        "south": ElementEntity.EDGE0,
        "north": ElementEntity.EDGE2
    }

    element_sets_tmp = {}
    for set_name in set_name_to_entity.keys():
        element_sets_tmp[set_name] = set()  #using sets temporarily to avoid duplicates

    for ey in range(nEy):
        for ex in range(nEx):
            e = IX_element(ex, ey)
            for il in range(nNl):
                i = (nNl_1D - 1) * ex + il_to_ij_loc[il][0]
                j = (nNl_1D - 1) * ey + il_to_ij_loc[il][1]
                element_sets_tmp["domain"].add(e)
                if i == 0:
                    element_sets_tmp["west"].add(e)
                if i == nNx - 1:
                    element_sets_tmp["east"].add(e)
                if j == 0:
                    element_sets_tmp["south"].add(e)
                if j == nNy - 1:
                    element_sets_tmp["north"].add(e)

    #Convert the sets to np arrays and store them in the element sets
    element_sets = {}
    for set_name, elem_set in element_sets_tmp.items():
        element_sets[set_name] = ElementSet(set_name_to_entity[set_name], np.array(sorted(elem_set), dtype=int))

    #Add the corner node sets
    node_sets = {}
    node_sets["south_west"] = np.array([IX_node(0, 0)], dtype=int)
    node_sets["south_east"] = np.array([IX_node(nNx - 1, 0)], dtype=int)
    node_sets["north_west"] = np.array([IX_node(0, nNy - 1)], dtype=int)
    node_sets["north_east"] = np.array([IX_node(nNx - 1, nNy - 1)], dtype=int)

    return Mesh(nodes, elements, node_sets, element_sets)


def perturb_mesh_nodes_of_structured_Q4_mesh(mesh_Q4: Mesh, Lx, Ly, nEx, nEy, perturb_mesh_nodes_factor):
    """Perturb the nodes of the structured mesh randomly to create badly shaped elements.
    This is useful to illustrate the effects of badly shaped elements.
    Note that the border nodes are constrained to stay on the original border."""
    if not 0 <= perturb_mesh_nodes_factor <= 1:
        raise ValueError(f"perturb_mesh_nodes_factor must be between 0 and 1, got {perturb_mesh_nodes_factor}")
    if perturb_mesh_nodes_factor == 0:
        return

    #Each node can move at most 0.3 element lengths, so that neighbouring nodes never cross each other.
    #(Close to 1, a few elements can still become invalid, see warn_about_invalid_elements.)
    SAFETY_FACTOR = 0.6
    nodes = mesh_Q4.nodes
    dx = Lx / nEx
    dy = Ly / nEy
    max_perturbation_dist_x = SAFETY_FACTOR * perturb_mesh_nodes_factor * 0.5 * dx
    max_perturbation_dist_y = SAFETY_FACTOR * perturb_mesh_nodes_factor * 0.5 * dy

    #Random number generator with a fixed seed, so that we get the same mesh every time the program runs
    rng = np.random.default_rng(seed=0)

    for i in range(nodes.shape[0]):
        factor_x, factor_y = rng.uniform(-1.0, 1.0, size=2)
        x, y = nodes[i, :]
        x_perturbed = x
        y_perturbed = y
        if SMALL_VAL < x < Lx - SMALL_VAL:
            #Node is neither on the left nor right border
            x_perturbed = x + factor_x * max_perturbation_dist_x
            assert 0 <= x_perturbed <= Lx
        if SMALL_VAL < y < Ly - SMALL_VAL:
            #Node is neither on the top nor bottom border
            y_perturbed = y + factor_y * max_perturbation_dist_y
            assert 0 <= y_perturbed <= Ly
        nodes[i, :] = np.array([x_perturbed, y_perturbed])


def warn_about_invalid_elements(mesh: Mesh):
    """Prints a warning if some elements are so distorted that they are not convex (or their nodes are numbered
    clockwise). Then det(J) <= 0 near a corner, which means that the element is not valid. (det(J) at a corner is
    proportional to the cross product of the two edges that meet there.)"""
    corners = mesh.nodes[mesh.elements[:, :4]]  #The first four nodes of an element are the corners, counter-clockwise
    invalid_elements = set()
    for k in range(4):
        edge_in = corners[:, k] - corners[:, k - 1]
        edge_out = corners[:, (k + 1) % 4] - corners[:, k]
        cross_product = edge_in[:, 0] * edge_out[:, 1] - edge_in[:, 1] * edge_out[:, 0]
        invalid_elements.update(np.where(cross_product <= 0)[0].tolist())
    if len(invalid_elements) > 0:
        print(f"WARNING: {len(invalid_elements)} element(s) are so distorted that they are not valid (det(J) <= 0 "
              f"near a corner): {sorted(invalid_elements)[:10]}. Check the positions of their nodes.")


def create_structured_quad_mesh(config: Config, Lx, Ly, nEx, nEy, perturb_mesh_nodes_factor=0) -> Mesh:
    """Creates a structured rectangular quad mesh with the specified parameters.
    Args:
        config (Config): Configuration object containing problem parameters.
        Lx (float): Length of the mesh in the x-direction.
        Ly (float): Length of the mesh in the y-direction.
        nEx (int): Number of elements in the x-direction.
        nEy (int): Number of elements in the y-direction.
        perturb_mesh_nodes_factor (float): Optional value between 0 and 1. If larger than 0, the interior
            nodes are moved randomly to create distorted elements. 1 gives the largest distortion.
    Returns:
        Mesh: A Mesh object containing the structured quad mesh.
    """
    if nEx < 1:
        raise ValueError(f"nEx must be at least 1, got {nEx}")
    if nEy < 1:
        raise ValueError(f"nEy must be at least 1, got {nEy}")
    if Lx < SMALL_VAL:
        raise ValueError(f"Lx must be greater than 0, got {Lx}")
    if Ly < SMALL_VAL:
        raise ValueError(f"Ly must be greater than 0, got {Ly}")

    mesh_Q4 = create_structured_Q4_mesh(Lx, Ly, nEx, nEy)
    perturb_mesh_nodes_of_structured_Q4_mesh(mesh_Q4, Lx, Ly, nEx, nEy, perturb_mesh_nodes_factor)
    warn_about_invalid_elements(mesh_Q4)
    mesh = create_higher_order_quad_mesh_from_Q4_mesh(mesh_Q4, config)
    return mesh
