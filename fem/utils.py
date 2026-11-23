from dataclasses import dataclass, field
from typing import Callable
from fem.fem_utils import *

#====================================================================
# The classes below are dataclasses: Simple containers of data (like
# a struct in C). Python creates the constructor automatically from
# the listed fields, e.g. BC(node_set_name, dof, value).
#====================================================================


@dataclass
class BC:
    """A boundary condition that prescribes a degree of freedom for all nodes in a node set,
    e.g. u = 0 for the node set "west". Created by add_boundary_condition()."""
    node_set_name: str  #Name of the node set that the boundary condition applies to
    dof: Dof  #The prescribed degree of freedom, e.g. Dof.U
    value: float  #The prescribed value


@dataclass
class Load:
    """A load on an element set. The element set defines if the load acts on edges (e.g. a traction or
    pressure) or on the face of the elements (e.g. a body force). Created by add_load()."""
    element_set_name: str  #Name of the element set that the load applies to
    load_type: LoadType  #E.g. LoadType.TRACTION
    load_function: Callable  #A function of (x, y) that returns the load, e.g. lambda x, y: (10, -1000 * x)


@dataclass
class Config:
    """Holds all the parameters and settings for the simulation and plotting. Created by create_config()."""

    #====================================================================
    # Problem settings
    #====================================================================
    E: float  #Young's modulus
    nu: float  #Poisson's ratio
    h: float  #Thickness
    element_type: ElementType
    problem_type: ProblemType  #Plane stress or plate problem
    hourglass_scaling: float = 0.1  #Scaling factor for hourglass stabilization term in Q4R element

    bcs: list[BC] = field(default_factory=list)  #List of boundary conditions (starts empty)
    loads: list[Load] = field(default_factory=list)  #List of loads (starts empty)

    #====================================================================
    # Plot settings
    #====================================================================
    show_mesh: bool = True
    disp_scaling: float = 1  #Scaling factor for displacements
    show_node_labels: bool = False
    show_bcs: bool = False
    contour: str = "u_mag"  #The field to show, e.g. "u_mag" or "sigma_xx" (see plot_tools.get_field_names)
    stress_recovery: str = "consistent"  #"consistent" or "L2" (see stress_recovery.py)
    arrow_type: str = "none"  #"none", "disp", "external forces" or "reaction forces"
    node_scale: float = 1  #Scales nodes and node labels
    arrow_scale: float = 1
    specify_contour_limits: bool = False
    contour_min: float = 0
    contour_max: float = 1


def create_config(E, nu, h, element_type, problem_type) -> Config:
    #ElementType(...) and ProblemType(...) give a clear error message if the value is not valid
    config = Config(E, nu, h, ElementType(element_type), ProblemType(problem_type))

    # Set problem-specific defaults
    if config.problem_type == ProblemType.PLATE:
        config.contour = "w"

    return config
