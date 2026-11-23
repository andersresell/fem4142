#====================================================================
# Everything that a script needs: from fem.useful_imports import *
# Only the names meant for scripts are imported here. The rest of the
# code can be imported directly when needed, e.g.
# from fem.fem_utils import ELEMENT_TRAITS
#====================================================================
import numpy as np
import matplotlib.pyplot as plt
from fem.fem_utils import ElementType, ProblemType, Dof, LoadType
from fem.utils import create_config
from fem.mesh import create_structured_quad_mesh
from fem.morph_mesh import morph_mesh_to_quarter_circle, morph_mesh_to_quarter_annulus, morph_mesh_to_quadrilateral
from fem.bcs import add_boundary_condition
from fem.loads import add_load
from fem.solver_data import create_solver_data, unpack_solution
from fem.solver import solve, calc_hourglass_energy
from fem.plot_tools import Plot, plot_field
import fem.stress_recovery as stress_recovery  #e.g. stress_recovery.calc_nodal_stress_L2
