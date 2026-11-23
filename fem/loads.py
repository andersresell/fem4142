from fem.mesh import Mesh
from fem.utils import Config, BC, Load
from fem.solver_data import SolverData
from fem.fem_utils import *
import fem.shape_functions as shape_functions

VALID_LOAD_FUNCTION_EXAMPLES = """
    Two examples of valid load functions are:
        Load with two components (e.g. traction): load_function = lambda x, y: (10, y)
        Load with one component (e.g. pressure):  load_function = lambda x, y: 10
    You may also write a normal function like:
        def load_function(x, y):
            return 10, y
"""


def validate_load_func(f):
    """Checks that the load function can be called as f(x, y), and returns the number of values it
    returns: 1 for a scalar load (e.g. pressure) or 2 for a vector load (e.g. traction)."""
    if not callable(f):
        raise TypeError(f"The load must be a function of x and y.\n{VALID_LOAD_FUNCTION_EXAMPLES}")
    try:
        result = f(0.0, 0.0)
    except TypeError as e:
        raise TypeError(f"Calling the load function as f(x=0.0, y=0.0) failed. It must take exactly two "
                        f"arguments (x, y). The error was: {e}\n{VALID_LOAD_FUNCTION_EXAMPLES}")
    except Exception as e:
        raise ValueError(f"The load function raised an error when called with (x, y) = (0.0, 0.0): {e}")

    try:
        result = np.asarray(result, dtype=float)
    except (TypeError, ValueError):
        raise TypeError(f"The load function must return one or two numbers, but it returned {result!r}.\n"
                        f"{VALID_LOAD_FUNCTION_EXAMPLES}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"The load function returned {result} at (x, y) = (0.0, 0.0). Did you forget 'return'?")
    if result.shape == ():
        return 1
    elif result.shape == (2,):
        return 2
    raise TypeError(f"The load function must return one or two numbers, but it returned {result.size} values.\n"
                    f"{VALID_LOAD_FUNCTION_EXAMPLES}")


def evaluate_scalar_load(load: Load, x, y):
    """Evaluates a load function that returns one value (e.g. a pressure) at the point (x, y)"""
    return float(load.load_function(x, y))


def evaluate_vector_load(load: Load, x, y):
    """Evaluates a load function that returns two values (e.g. a traction) at the point (x, y)
    and returns them as a numpy array, so that we can calculate with it."""
    value = np.asarray(load.load_function(x, y), dtype=float)
    assert value.shape == (2,)
    return value


def check_load_validity(config: Config, mesh: Mesh, element_set_name, load_type, load_function):

    element_sets = mesh.element_sets

    #====================================================================
    # First some checks to ensure that the element set exists
    #====================================================================
    if element_set_name not in element_sets:
        all_element_set_names = ", ".join(
            element_sets.keys())  #Find all available element set names to display in the error message
        raise Exception(f"Element set '{element_set_name}' not found. Available: {all_element_set_names}")

    #====================================================================
    # Then some checks to ensure that the load type is valid for
    # the problem type and the type of element set
    #====================================================================
    element_set = element_sets[element_set_name]
    is_face = element_set.element_entity == ElementEntity.FACE  #Otherwise, the element set defines edges
    problem_type_name = config.problem_type.name
    if config.problem_type == ProblemType.PLANE_STRESS:
        if not is_face:
            if load_type not in [LoadType.TRACTION, LoadType.PRESSURE]:
                raise Exception(f"LoadType.{load_type.name} is not valid for {problem_type_name} "
                                f"when the element set defines an edge.")
        else:
            if load_type not in [LoadType.TRACTION, LoadType.BODY_FORCE]:
                raise Exception(f"LoadType.{load_type.name} is not valid for {problem_type_name} "
                                f"when the element set defines a face.")
    else:
        assert config.problem_type == ProblemType.PLATE
        if not is_face:
            if load_type != LoadType.TRACTION:
                raise Exception(f"Only LoadType.TRACTION is valid for {problem_type_name} "
                                f"when the element set defines an edge.")
        else:
            if load_type not in [LoadType.PRESSURE, LoadType.BODY_FORCE]:
                raise Exception(f"Only LoadType.PRESSURE and LoadType.BODY_FORCE are valid for {problem_type_name} "
                                f"when the element set defines a face.")

    #====================================================================
    # Then a bunch of checks to ensure the load function is valid
    #====================================================================
    num_output_components = validate_load_func(load_function)
    assert num_output_components in [1, 2]
    if config.problem_type == ProblemType.PLANE_STRESS:
        if load_type == LoadType.TRACTION or load_type == LoadType.BODY_FORCE:
            if num_output_components == 1:
                raise Exception(
                    f"A load function that returns a single value cannot be used with LoadType.TRACTION or LoadType.BODY_FORCE in "
                    f"{problem_type_name}. Use LoadType.PRESSURE instead.")
            elif num_output_components != 2:
                raise Exception(f"Load function must return either a single value or a tuple of two values. "
                                f"Got {num_output_components} components instead.")
        else:
            assert load_type == LoadType.PRESSURE
            if num_output_components != 1:
                raise Exception(f"Load function must return a single value for LoadType.PRESSURE in {problem_type_name}. "
                                f"Got {num_output_components} components instead.")
    else:
        assert config.problem_type == ProblemType.PLATE
        if num_output_components != 1:
            raise Exception(f"LoadType.{load_type.name} is not valid for a load function that returns "
                            f"{num_output_components} components in {problem_type_name}. "
                            f"Use a single-valued load function for LoadType.TRACTION, LoadType.PRESSURE or "
                            f"LoadType.BODY_FORCE.")


def add_load(config: Config, mesh: Mesh, element_set_name, load_type, load_function):
    """Adds a load to the specified element set with the given load function.
       The load function is a lambda or function that takes (x, y) coordinates and
       returns either a single value (for LoadType.PRESSURE) or a tuple of two values
       (for LoadType.TRACTION or LoadType.BODY_FORCE).
       Examples:
       load_function = lambda x, y: (10, x) # For LoadType.TRACTION or LoadType.BODY_FORCE
       load_function = lambda x, y: 10 # For LoadType.PRESSURE
    """
    load_type = LoadType(load_type)  #Gives a clear error message if the load type is not valid
    try:
        check_load_validity(config, mesh, element_set_name, load_type, load_function)
    except Exception as e:
        raise ValueError(f"Invalid load configuration for element set '{element_set_name}' "
                         f"and LoadType.{load_type.name}:\n{e}")

    config.loads.append(Load(element_set_name, load_type, load_function))


def integrate_element_face_load_plane_stress(config: Config, mesh: Mesh, load: Load, nodeIDs_l, e):
    assert config.problem_type == ProblemType.PLANE_STRESS
    element_type = config.element_type
    h = config.h
    nNl_face = len(nodeIDs_l)
    assert nNl_face == ELEMENT_TRAITS[config.element_type].nNl
    nodeIDs = mesh.elements[e, nodeIDs_l]
    x_l = mesh.nodes[nodeIDs, 0]
    y_l = mesh.nodes[nodeIDs, 1]
    #Loads are always fully integrated (as many Gauss points as nodes along an edge), also for R elements
    nGauss_1D = ELEMENT_TRAITS[element_type].nNl_1D
    arr_xi = shape_functions.get_arr_xi(nGauss_1D)
    arr_w = shape_functions.get_arr_w(nGauss_1D)
    R_load = np.zeros(2 * nNl_face)
    for i in range(nGauss_1D):
        for j in range(nGauss_1D):
            xi = arr_xi[i]
            eta = arr_xi[j]
            w = arr_w[i] * arr_w[j]
            N = shape_functions.calc_N(xi, eta, element_type)
            J = shape_functions.calc_J(xi, eta, x_l, y_l, element_type)
            detJ = np.linalg.det(J)
            if detJ <= 0:
                raise ValueError(f"Element {e} has a zero or negative Jacobian determinant: {detJ}. "
                                 f"Check the element geometry and node ordering.")

            #Used to evaluate the load function at the Gauss points
            x_Gauss = N @ x_l
            y_Gauss = N @ y_l

            if load.load_type == LoadType.TRACTION:
                t = evaluate_vector_load(load, x_Gauss, y_Gauss)
            else:
                assert load.load_type == LoadType.BODY_FORCE
                b = evaluate_vector_load(load, x_Gauss, y_Gauss)
                t = h * b  #convert body force to a in plane traction by multiplying by the plate thickness

            R_load[0::2] += t[0] * N * detJ * w
            R_load[1::2] += t[1] * N * detJ * w
    return R_load


def integrate_element_edge_load_plane_stress(config: Config, mesh: Mesh, load: Load, nodeIDs_l, e):
    assert config.problem_type == ProblemType.PLANE_STRESS
    element_type = config.element_type
    h = config.h
    nNl_edge = len(nodeIDs_l)
    nodeIDs = mesh.elements[e, nodeIDs_l]
    x_edge = mesh.nodes[nodeIDs, 0]
    y_edge = mesh.nodes[nodeIDs, 1]
    #Loads are always fully integrated (as many Gauss points as nodes along an edge), also for R elements
    nGauss = ELEMENT_TRAITS[element_type].nNl_1D
    arr_xi = shape_functions.get_arr_xi(nGauss)
    arr_w = shape_functions.get_arr_w(nGauss)
    R_load = np.zeros(2 * nNl_edge)
    for i in range(nGauss):
        xi = arr_xi[i]
        w = arr_w[i]
        N = shape_functions.calc_N_1D(xi, nNl_edge)
        dNdxi = shape_functions.calc_dNdxi_1D(xi, nNl_edge)
        dxdxi = dNdxi @ x_edge
        dydxi = dNdxi @ y_edge
        detJ = np.sqrt(dxdxi**2 + dydxi**2)
        if detJ <= 0:
            raise ValueError(f"Element {e} has a zero or negative Jacobian determinant: {detJ}. "
                             f"Check the element geometry and node ordering.")

        # Normal vector to the edge (the edge is assumed to run counter-clockwise around the element)
        normal = np.array([dydxi, -dxdxi]) / detJ

        #Used to evaluate the load function at the Gauss points
        x_Gauss = N @ x_edge
        y_Gauss = N @ y_edge
        if load.load_type == LoadType.TRACTION:
            t = evaluate_vector_load(load, x_Gauss, y_Gauss)
        else:
            assert load.load_type == LoadType.PRESSURE
            p = evaluate_scalar_load(load, x_Gauss, y_Gauss)
            t = -p * normal

        R_load[0::2] += t[0] * N * detJ * w * h
        R_load[1::2] += t[1] * N * detJ * w * h
    return R_load


def integrate_element_face_load_plate(config: Config, mesh: Mesh, load: Load, nodeIDs_l, e):
    assert config.problem_type == ProblemType.PLATE
    element_type = config.element_type
    h = config.h
    nNl_face = len(nodeIDs_l)
    assert nNl_face == ELEMENT_TRAITS[config.element_type].nNl
    nodeIDs = mesh.elements[e, nodeIDs_l]
    x_l = mesh.nodes[nodeIDs, 0]
    y_l = mesh.nodes[nodeIDs, 1]
    #Loads are always fully integrated (as many Gauss points as nodes along an edge), also for R elements
    nGauss_1D = ELEMENT_TRAITS[element_type].nNl_1D
    arr_xi = shape_functions.get_arr_xi(nGauss_1D)
    arr_w = shape_functions.get_arr_w(nGauss_1D)
    #We make the load vector contain all 3 dofs w,thetax, thetay, even though we only fill the w dofs
    #This is because we don't want special code in the assembly procedure dealing with a reduced form of the shape functions
    R_load = np.zeros(3 * nNl_face)
    for i in range(nGauss_1D):
        for j in range(nGauss_1D):
            xi = arr_xi[i]
            eta = arr_xi[j]
            w = arr_w[i] * arr_w[j]
            N = shape_functions.calc_N(xi, eta, element_type)
            J = shape_functions.calc_J(xi, eta, x_l, y_l, element_type)
            detJ = np.linalg.det(J)
            if detJ <= 0:
                raise ValueError(f"Element {e} has a zero or negative Jacobian determinant: {detJ}. "
                                 f"Check the element geometry and node ordering.")

            #Used to evaluate the load function at the Gauss points
            x_Gauss = N @ x_l
            y_Gauss = N @ y_l

            if load.load_type == LoadType.PRESSURE:
                p = evaluate_scalar_load(load, x_Gauss, y_Gauss)
            else:
                assert load.load_type == LoadType.BODY_FORCE
                b = evaluate_scalar_load(load, x_Gauss, y_Gauss)
                p = h * b  #We convert the body force to a pressure by multiplying with thickness

            R_load[0::3] += -p * N * detJ * w  #The pressure force only contributes to the w-dofs

    return R_load


def integrate_element_edge_load_plate(config: Config, mesh: Mesh, load: Load, nodeIDs_l, e):
    assert config.problem_type == ProblemType.PLATE
    element_type = config.element_type
    h = config.h
    nNl_edge = len(nodeIDs_l)
    nodeIDs = mesh.elements[e, nodeIDs_l]
    x_edge = mesh.nodes[nodeIDs, 0]
    y_edge = mesh.nodes[nodeIDs, 1]
    #Loads are always fully integrated (as many Gauss points as nodes along an edge), also for R elements
    nGauss = ELEMENT_TRAITS[element_type].nNl_1D
    arr_xi = shape_functions.get_arr_xi(nGauss)
    arr_w = shape_functions.get_arr_w(nGauss)
    #We specify an empty load vector containing all dofs w,thetax,thetaz, allthough the two latter will remain zero
    R_load = np.zeros(3 * nNl_edge)
    for i in range(nGauss):
        xi = arr_xi[i]
        w = arr_w[i]
        N = shape_functions.calc_N_1D(xi, nNl_edge)
        dNdxi = shape_functions.calc_dNdxi_1D(xi, nNl_edge)
        dxdxi = dNdxi @ x_edge
        dydxi = dNdxi @ y_edge
        detJ = np.sqrt(dxdxi**2 + dydxi**2)
        if detJ <= 0:
            raise ValueError(f"Element {e} has a zero or negative Jacobian determinant: {detJ}. "
                             f"Check the element geometry and node ordering.")

        #Used to evaluate the load function at the Gauss points
        x_Gauss = N @ x_edge
        y_Gauss = N @ y_edge

        #We only accept traction loads for the edge loads of the plates.
        #However this traction function should return a single value.
        #In global 3D space we assume the traction has the form t = t_z * [0, 0, 1]^T, and we only provide t_z
        assert load.load_type == LoadType.TRACTION
        t_z = evaluate_scalar_load(load, x_Gauss, y_Gauss)

        R_load[0::3] += t_z * N * detJ * w * h  #This traction load only acts in the z direction
    return R_load


def integrate_element_load(config: Config, mesh: Mesh, load: Load, nodeIDs_l, e, element_entity):
    if config.problem_type == ProblemType.PLANE_STRESS:
        if element_entity == ElementEntity.FACE:
            R_load = integrate_element_face_load_plane_stress(config, mesh, load, nodeIDs_l, e)
        else:
            R_load = integrate_element_edge_load_plane_stress(config, mesh, load, nodeIDs_l, e)
    else:
        assert config.problem_type == ProblemType.PLATE
        if element_entity == ElementEntity.FACE:
            R_load = integrate_element_face_load_plate(config, mesh, load, nodeIDs_l, e)
        else:
            R_load = integrate_element_edge_load_plate(config, mesh, load, nodeIDs_l, e)

    return R_load


def assemble_loads_consistent(config: Config, mesh: Mesh, solver_data: SolverData):

    element_sets = mesh.element_sets
    elements = mesh.elements
    element_traits = ELEMENT_TRAITS[config.element_type]

    for load in config.loads:
        element_set = element_sets[load.element_set_name]
        element_entity = element_set.element_entity
        nodeIDs_l = element_traits.entity_nodes[element_entity]  #The element nodes on the edge or face
        nNl_ent = len(nodeIDs_l)
        NUM_DOFS = get_num_dofs_from_problem_type(config.problem_type)

        for e in element_set.elementIDs:

            #====================================================================
            # Integrate consistent load for the element in question
            #====================================================================
            R_load = integrate_element_load(config, mesh, load, nodeIDs_l, e, element_entity)

            #====================================================================
            # Assemble contribution into the external force vector R_ext
            #====================================================================
            assert len(R_load) == nNl_ent * NUM_DOFS
            nodeIDs = elements[e, nodeIDs_l]
            for il in range(nNl_ent):
                for dof in range(NUM_DOFS):
                    I = nodeIDs[il] * NUM_DOFS + dof
                    solver_data.R_ext[I] += R_load[NUM_DOFS * il + dof]

    print("Consistent load integration complete")
