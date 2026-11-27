# Documentation of the educational 2D FEM tool

How to set up and solve problems with the `fem` package, and what the names in the code mean.

**Contents**

1. [Requirements and editor](#requirements-and-editor)
2. [Getting started](#getting-started)
3. [Example: stresses over a cross-section](#example-stresses-over-a-cross-section)
4. [Example: Mindlin plate](#example-mindlin-plate)
5. [Theory and conventions](#theory-and-conventions)
6. [Element types](#element-types)
7. [Config](#config)
8. [Mesh](#mesh)
9. [Boundary conditions](#boundary-conditions)
10. [Loads](#loads)
11. [Solver data and results](#solver-data-and-results)
12. [Stresses](#stresses)
13. [Plotting](#plotting)
14. [Short names in the code](#short-names-in-the-code)

## Requirements and editor

The code needs Python 3.10 or newer with the packages numpy, scipy and matplotlib. It is tested with Python 3.10
to 3.13, numpy 2.2 to 2.5, scipy 1.15 to 1.18 and matplotlib 3.10 and 3.11. Other versions of the packages will
probably work as well, but only the tested versions are guaranteed to work.

We recommend writing and running the scripts in Visual Studio Code with the Python extension from Microsoft
(search for "Python" under Extensions). It gives syntax highlighting, and shows the arguments and descriptions of
the functions while you type. Run a script with the Run button in the top right corner of the editor. If VS Code
can't find numpy or the other packages, choose the Python installation that has them with the command
*Python: Select Interpreter* (Ctrl+Shift+P opens the list of commands).

The GUI is interactive, so run the scripts as normal Python scripts, e.g. in VS Code or from a terminal. The GUI
doesn't work in Jupyter notebooks. In Spyder, set *Tools > Preferences > IPython console > Graphics > Backend* to
*Automatic* and restart the console, since the plots are otherwise shown as static images.

## Getting started

Every script follows the same steps. This one solves a cantilever under a uniform pressure and opens the GUI:

```python
from fem.useful_imports import *  #numpy (np), matplotlib (plt) and the functions below

config = create_config(E=210e9, nu=0.3, h=0.01, element_type=ElementType.Q9,
                       problem_type=ProblemType.PLANE_STRESS)
mesh = create_structured_quad_mesh(config, Lx=2.0, Ly=0.2, nEx=20, nEy=2)  #A rectangle
add_boundary_condition(config, mesh, "west", Dof.U, 0)  #u = 0 on the left edge
add_boundary_condition(config, mesh, "west", Dof.V, 0)  #v = 0 on the left edge
add_load(config, mesh, "north", LoadType.PRESSURE, 1e5)  #Pressure on the top edge
solver_data = create_solver_data(config, mesh)  #The system vectors and matrices
solve(config, solver_data, mesh)  #Assembles and solves K r = R_ext
GUI(config, mesh, solver_data)  #Opens the GUI
```

The code doesn't use units, so use a consistent set, e.g. N, m and Pa.

**Where to put your scripts.** Python must be able to find the folder `fem`. A script in the project folder (the
folder that contains `fem`) works as it is. To keep your scripts in a subfolder, e.g.

```
fem4142/
├── fem/
└── my_scripts/
    └── my_case.py
```

start them with these lines, which add the folder above the script to the places where Python looks for code:

```python
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from fem.useful_imports import *
```

## Example: stresses over a cross-section

A cantilever of length $L$ and height $H$, clamped at the left edge, with a downward force $P$ at the right edge.
The stress $\sigma_{xx}$ over the cross-section $x = L/2$ is taken from the nodal stresses (L2 projection, see
[Stresses](#stresses)) and compared with beam theory, $\sigma_{xx} = M (y - H/2) / I$ with $M = P(L - x)$ and
$I = hH^3/12$.

```python
from fem.useful_imports import *

E, nu, h, L, H, P = 210e9, 0.3, 0.01, 2.0, 0.2, 1000.0
config = create_config(E, nu, h, ElementType.Q9, ProblemType.PLANE_STRESS)
mesh = create_structured_quad_mesh(config, L, H, 20, 4)
add_boundary_condition(config, mesh, "west", Dof.U, 0)
add_boundary_condition(config, mesh, "west", Dof.V, 0)
add_load(config, mesh, "east", LoadType.TRACTION, (0, -P / (H * h)))  #Force per area
solver_data = create_solver_data(config, mesh)
solve(config, solver_data, mesh)

#The nodes on the cross-section x = L/2, sorted by y
x, y = mesh.nodes[:, 0], mesh.nodes[:, 1]
section = np.where(np.isclose(x, L / 2))[0]
section = section[np.argsort(y[section])]
#The stresses at the nodes. Columns: sigma_xx, sigma_yy, sigma_xy
nodal_stress = stress_recovery.calc_nodal_stress_L2(config, mesh, solver_data.r)

I = h * H**3 / 12
plt.plot(nodal_stress[section, 0], y[section], 'o', label="FEM")
plt.plot(P * (L / 2) * (y[section] - H / 2) / I, y[section], label="Beam theory")
plt.xlabel(r"$\sigma_{xx}$")
plt.ylabel(r"$y$")
plt.legend()
plt.show()
```

## Example: Mindlin plate

A simply supported square plate with side length $L$ under a uniform pressure $p$. The center deflection is
compared with Kirchhoff (thin plate) theory, $w_c = -0.00406 \cdot pL^4/D$ with $D = Eh^3/(12(1-\nu^2))$.

```python
from fem.useful_imports import *

E, nu, h, L, p = 210e9, 0.3, 0.01, 1.0, 1000.0
config = create_config(E, nu, h, ElementType.Q9, ProblemType.PLATE)
mesh = create_structured_quad_mesh(config, L, L, 8, 8)
for edge in ["west", "east", "south", "north"]:
    add_boundary_condition(config, mesh, edge, Dof.W, 0)  #Simply supported
add_load(config, mesh, "domain", LoadType.PRESSURE, p)  #Positive downwards
solver_data = create_solver_data(config, mesh)
solve(config, solver_data, mesh)

w, theta_x, theta_y = unpack_solution(config, solver_data.r)
#The node closest to the center
center = np.argmin((mesh.nodes[:, 0] - L / 2)**2 + (mesh.nodes[:, 1] - L / 2)**2)
D = E * h**3 / (12 * (1 - nu**2))
w_kirchhoff = -0.00406 * p * L**4 / D
print(f"Center deflection: {w[center]:.4e}. Kirchhoff theory: {w_kirchhoff:.4e}")

config.contour = "M_xx"
GUI(config, mesh, solver_data)
```

## Theory and conventions

In each element, the unknowns are interpolated from the $n$ nodes with the shape functions $N_k(\xi, \eta)$, e.g.
$u = \sum_k N_k u_k$. The unknowns of the element are collected in the vector $\mathbf{r}_e$.

**Plane stress.** A thin body with thickness $h$, loaded in its plane. The unknowns are the displacements $u$ and
$v$, so $\mathbf{r}_e = [u_1 \ v_1 \ \cdots \ u_n \ v_n]^T$. The strains and stresses are

```math
\boldsymbol{\varepsilon} = \begin{bmatrix} \varepsilon_{xx} \\ \varepsilon_{yy} \\ \gamma_{xy} \end{bmatrix}
= \begin{bmatrix} \partial u / \partial x \\ \partial v / \partial y \\ \partial u / \partial y + \partial v / \partial x \end{bmatrix}
= \mathbf{B} \mathbf{r}_e, \qquad
\mathbf{B}_k = \begin{bmatrix} \partial N_k / \partial x & 0 \\ 0 & \partial N_k / \partial y \\ \partial N_k / \partial y & \partial N_k / \partial x \end{bmatrix}
```

```math
\boldsymbol{\sigma} = \begin{bmatrix} \sigma_{xx} \\ \sigma_{yy} \\ \sigma_{xy} \end{bmatrix}
= \mathbf{D} \boldsymbol{\varepsilon} = \mathbf{D} \mathbf{B} \mathbf{r}_e, \qquad
\mathbf{D} = \frac{E}{1-\nu^2} \begin{bmatrix} 1 & \nu & 0 \\ \nu & 1 & 0 \\ 0 & 0 & \frac{1-\nu}{2} \end{bmatrix}
```

where $\mathbf{B} = [\mathbf{B}_1 \ \cdots \ \mathbf{B}_n]$ has one block per node. The element stiffness matrix is

```math
\mathbf{K}_e = \int_{A_e} \mathbf{B}^T \mathbf{D} \mathbf{B} \, h \, dA
```

**Mindlin plate.** The unknowns are the deflection $w$ and the rotations $\theta_x$ and $\theta_y$ about the $x$- and
$y$-axis, so $\mathbf{r}_e = [w_1 \ \theta_{x1} \ \theta_{y1} \ \cdots \ w_n \ \theta_{xn} \ \theta_{yn}]^T$. The
$z$-axis points upwards, and the mid-plane is at $z = 0$. The rotations follow the right-hand rule, so a point at
height $z$ moves $u = z \theta_y$ and $v = -z \theta_x$ in the plane. The curvatures $\boldsymbol{\kappa}$ and the
shear strains $\boldsymbol{\gamma}$ are

```math
\begin{gathered}
\boldsymbol{\kappa} = \begin{bmatrix} \kappa_{xx} \\ \kappa_{yy} \\ \kappa_{xy} \end{bmatrix}
= \begin{bmatrix} -\partial \theta_y / \partial x \\ \partial \theta_x / \partial y \\ \partial \theta_x / \partial x - \partial \theta_y / \partial y \end{bmatrix}
= \mathbf{B}_b \mathbf{r}_e, \qquad
\boldsymbol{\gamma} = \begin{bmatrix} \gamma_{xz} \\ \gamma_{yz} \end{bmatrix}
= \begin{bmatrix} \partial w / \partial x + \theta_y \\ \partial w / \partial y - \theta_x \end{bmatrix}
= \mathbf{B}_s \mathbf{r}_e \\[6pt]
\mathbf{B}_{b,k} = \begin{bmatrix} 0 & 0 & -\partial N_k / \partial x \\ 0 & \partial N_k / \partial y & 0 \\ 0 & \partial N_k / \partial x & -\partial N_k / \partial y \end{bmatrix}, \qquad
\mathbf{B}_{s,k} = \begin{bmatrix} \partial N_k / \partial x & 0 & N_k \\ \partial N_k / \partial y & -N_k & 0 \end{bmatrix}
\end{gathered}
```

The moments $\mathbf{M}$ and shear forces $\mathbf{V}$ (per unit length) are

```math
\begin{gathered}
\mathbf{M} = \begin{bmatrix} M_{xx} \\ M_{yy} \\ M_{xy} \end{bmatrix} = \mathbf{D}_b \boldsymbol{\kappa}, \qquad
\mathbf{V} = \begin{bmatrix} V_x \\ V_y \end{bmatrix} = \mathbf{D}_s \boldsymbol{\gamma} \\[6pt]
\mathbf{D}_b = \frac{Eh^3}{12(1-\nu^2)} \begin{bmatrix} 1 & \nu & 0 \\ \nu & 1 & 0 \\ 0 & 0 & \frac{1-\nu}{2} \end{bmatrix}, \qquad
\mathbf{D}_s = \frac{5}{6} G h \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}, \qquad G = \frac{E}{2(1+\nu)}
\end{gathered}
```

and the element stiffness matrix is

```math
\mathbf{K}_e = \int_{A_e} \left( \mathbf{B}_b^T \mathbf{D}_b \mathbf{B}_b + \mathbf{B}_s^T \mathbf{D}_s \mathbf{B}_s \right) dA
```

A positive moment gives tension at the bottom of the plate ($z = -h/2$). A positive pressure acts downwards and
gives a negative $w$.

**Integration and assembly.** The integrals are calculated with Gauss quadrature in the element's own coordinates
$-1 \le \xi, \eta \le 1$, e.g. $\mathbf{K}_e \approx \sum_i \sum_j \mathbf{B}^T \mathbf{D} \mathbf{B} \ h \det \mathbf{J} \ w_i w_j$,
where $\mathbf{J}$ is the Jacobian matrix of the mapping from $(\xi, \eta)$ to $(x, y)$ and $w_i$ are the Gauss
weights. The element matrices are assembled into the global stiffness matrix $\mathbf{K}$, the loads give the
external force vector $\mathbf{R}_{\text{ext}}$, and the system $\mathbf{K}\mathbf{r} = \mathbf{R}_{\text{ext}}$ is
solved with the boundary conditions applied.

## Element types

All elements are quadrilaterals. R means reduced integration: one Gauss point less in each direction. This removes
most of the shear locking, but may allow spurious zero-energy modes (deformations without strain energy). If the
system becomes singular, `solve()` prints a warning, and the results should not be trusted.

| Element | Nodes | Gauss points | Comments |
|--------|------|--------|------------------------------------------------------------|
| `Q4` | 4 | 2 × 2 | Bilinear. Much too stiff in bending (shear locking), especially on coarse or skewed meshes. |
| `Q4R` | 4 | 1 × 1 | For plane stress, an artificial hourglass stiffness prevents spurious modes. It is scaled by `config.hourglass_scaling`. |
| `Q8` | 8 | 3 × 3 | Quadratic, without a center node (serendipity). |
| `Q8R` | 8 | 2 × 2 | A good element for plane stress, although it has no hourglass control: its spurious mode only exists in a single element, and disappears in a mesh of two or more elements. Not recommended for plates, since thin plates still lock in shear. |
| `Q9` | 9 | 3 × 3 | Quadratic (Lagrange). |
| `Q9R` | 9 | 2 × 2 | Has spurious modes that also exist in larger meshes. Depending on the boundary conditions, the system may be singular. |
| `Q16` | 16 | 4 × 4 | Cubic (Lagrange). The most accurate. |
| `Q4_USER` | 4 | 2 × 2 | For exercises: the stiffness matrix is written by the student in `fem/element_stiffness_user.py`. |

In each element, the first four nodes are the corners (counter-clockwise), followed by the nodes on the edges and
the interior nodes.

## Config

`config = create_config(E, nu, h, element_type, problem_type)` collects the settings of a problem. All fields can be
changed after the config is created, e.g. `config.hourglass_scaling = 0.01` or `config.contour = "sigma_vm"`. The
plot settings give the starting state of the GUI.

| Field | Meaning |
|----------------------------|------------------------------------------------|
| `E`, `nu` | Young's modulus $E$ and Poisson's ratio $\nu$ |
| `h` | Thickness $h$ (out of the plane for plane stress) |
| `element_type` | `ElementType.Q4`, `ElementType.Q9`, ... (see [Element types](#element-types)) |
| `problem_type` | `ProblemType.PLANE_STRESS` or `ProblemType.PLATE` |
| `hourglass_scaling` | Scales the hourglass stiffness of `Q4R` in plane stress (default 0.1) |
| `bcs`, `loads` | Lists of boundary conditions and loads, filled by `add_boundary_condition` and `add_load` |
| `contour` | The field shown as contours (see [Plotting](#plotting)) |
| `stress_recovery` | `"consistent"` or `"L2"` (see [Stresses](#stresses)) |
| `arrow_type` | `"none"`, `"disp"`, `"external forces"` or `"reaction forces"` (plane stress only) |
| `show_labels`, `show_nodes`, `show_element_edges`, `show_bcs`, `show_deformed` | What is drawn (`True`/`False`) |
| `disp_scaling` | Factor for the displacements in the deformed plot. `"auto"`: the largest is 10% of the mesh size |
| `symbol_scale`, `arrow_scale` | Size of the symbols and arrows |
| `specify_contour_limits`, `contour_min`, `contour_max` | Fixed limits for the contours |

## Mesh

`mesh = create_structured_quad_mesh(config, Lx, Ly, nEx, nEy)` creates a rectangle $[0, L_x] \times [0, L_y]$ with
`nEx` × `nEy` elements. The optional `perturb_mesh_nodes_factor` (between 0 and 1) moves the interior nodes randomly,
which gives distorted elements.

| Field | Meaning |
|-----------------------|-----------------------------------------------------|
| `mesh.nodes` | The node coordinates, one row $(x, y)$ per node |
| `mesh.elements` | The node numbers of each element, one row per element |
| `mesh.node_sets` | Named sets of node numbers, used for boundary conditions |
| `mesh.element_sets` | Named sets of element edges or faces, used for loads |
| `mesh.get_nN()`, `mesh.get_nE()` | The number of nodes and elements |

The sets are named after the compass directions:

```
                    north
  north_west .________________. north_east
             |                |
        west |     domain     | east
             |                |
  south_west .________________. south_east
                    south
```

The node sets are `"domain"` (all nodes), the four edges and the four corners. The element sets are `"domain"` (the
faces of all elements) and the four edges.

The mesh can be morphed into another shape. The sets keep their names.

| Function | Shape |
|------------------------------------|--------------------------------------------|
| `morph_mesh_to_quarter_circle(mesh, radius)` | A quarter circle centered at the origin. `"south"` lies along the $x$-axis, `"west"` along the $y$-axis, and `"east"` and `"north"` form the arc. |
| `morph_mesh_to_quarter_annulus(mesh, inner_radius, outer_radius)` | A quarter ring. `"west"` is the inner arc and `"east"` the outer arc. `"south"` lies along the $x$-axis and `"north"` along the $y$-axis. |
| `morph_mesh_to_quadrilateral(mesh, corners)` | A convex quadrilateral with straight edges. `corners` are the south-west, south-east, north-east and north-west corners (counter-clockwise). |

## Boundary conditions

`add_boundary_condition(config, mesh, node_set_name, dof, value=0)` prescribes the dof to `value` at all nodes in
the node set. A non-zero value gives a prescribed displacement or rotation. Enough dofs must be prescribed to
prevent rigid body motion, otherwise `solve()` warns that the system is singular.

| Condition | Plane stress | Plate |
|------------------------------|--------------------|---------------------------|
| Clamped | `Dof.U` and `Dof.V` | `Dof.W`, `Dof.THETAX` and `Dof.THETAY` |
| Simply supported | | `Dof.W` |
| Symmetry about a line $x$ = constant | `Dof.U` | `Dof.THETAY` |
| Symmetry about a line $y$ = constant | `Dof.V` | `Dof.THETAX` |

## Loads

`add_load(config, mesh, element_set_name, load_type, load_function)` adds a load on an element set. The load has
either one value or two values $(f_x, f_y)$ (see the table below), and is given in one of two ways:

- A constant: a number, e.g. `1000`, or two numbers, e.g. `(0, -9810)`.
- A function of the position $(x, y)$ that returns the value(s), e.g. `lambda x, y: 1000 * x` or
  `lambda x, y: (0, -9810 * y)`.

The loads are integrated consistently with the shape functions, $\mathbf{R}_e = \int \mathbf{N}^T \mathbf{t} \ dA$.
Several loads add up.

| Problem | Element set | Load type | Returns | Meaning |
|----------|----------|-------------|---------|----------------------------------------|
| Plane stress | An edge | `TRACTION` | $(t_x, t_y)$ | Force per area of the edge. The force per length is $h \mathbf{t}$. |
| Plane stress | An edge | `PRESSURE` | $p$ | Force per area, normal to the edge. Positive pushes into the body: $\mathbf{t} = -p \mathbf{n}$, where $\mathbf{n}$ is the outward normal. |
| Plane stress | `"domain"` | `BODY_FORCE` | $(b_x, b_y)$ | Force per volume, e.g. gravity $(0, -\rho g)$ |
| Plane stress | `"domain"` | `TRACTION` | $(t_x, t_y)$ | Force per area of the face |
| Plate | `"domain"` | `PRESSURE` | $p$ | Force per area. Positive acts downwards. |
| Plate | `"domain"` | `BODY_FORCE` | $b$ | Force per volume, e.g. the weight $\rho g$. Positive acts downwards. |
| Plate | An edge | `TRACTION` | $t_z$ | Force per area of the edge. Positive acts upwards. The force per length is $h t_z$. |

## Solver data and results

`solver_data = create_solver_data(config, mesh)` holds the system $\mathbf{K}\mathbf{r} = \mathbf{R}_{\text{ext}}$,
and `solve(config, solver_data, mesh)` assembles and solves it.

| Field | Meaning |
|--------------------------|--------------------------------------------------|
| `solver_data.r` | The solution $\mathbf{r}$: all dofs, node by node, $(u_0, v_0, u_1, v_1, \ldots)$ or $(w_0, \theta_{x0}, \theta_{y0}, w_1, \ldots)$ |
| `solver_data.R_ext` | The external forces $\mathbf{R}_{\text{ext}}$ from the loads |
| `solver_data.R_int` | The internal forces $\mathbf{R}_{\text{int}} = \mathbf{K}\mathbf{r}$ |
| `solver_data.K` | The stiffness matrix $\mathbf{K}$ (sparse) |
| `solver_data.A`, `solver_data.b` | $\mathbf{K}$ and $\mathbf{R}_{\text{ext}}$ with the boundary conditions applied: the system that is actually solved |

`unpack_solution(config, vector)` splits a vector like $\mathbf{r}$ into one array per dof, with one value per node:
`u, v = unpack_solution(config, solver_data.r)` for plane stress, and
`w, theta_x, theta_y = unpack_solution(config, solver_data.r)` for plates.

The reaction forces at the prescribed dofs are $\mathbf{R}_{\text{int}} - \mathbf{R}_{\text{ext}}$.

`U_hg, U = calc_hourglass_energy(config, mesh, solver_data)` gives the strain energy in the artificial hourglass
stiffness of `Q4R`, and the total strain energy. A large $U_{hg}/U$ means that the hourglass stiffness affects the
results.

## Stresses

The stresses are calculated from the solution in two ways:

- **Consistent**: $\boldsymbol{\sigma} = \mathbf{D}\mathbf{B}\mathbf{r}_e$ in each element. This jumps between elements.
- **L2**: Continuous nodal stresses, found by a least squares fit of the consistent stresses (an L2 projection).

Both functions below return an array with one row per point and one column per stress component:
$\sigma_{xx}, \sigma_{yy}, \sigma_{xy}$ for plane stress, and $M_{xx}, M_{yy}, M_{xy}, V_x, V_y$ for plates.

| Function | Returns |
|------------------------------------------|----------------------------------|
| `stress_recovery.calc_nodal_stress_L2(config, mesh, solver_data.r)` | The L2 stresses at the nodes (one row per node) |
| `stress_recovery.calc_stress_consistent(config, mesh, solver_data.r, e, xi_eta)` | The consistent stresses in element `e`, at the points `xi_eta` $= [(\xi_0, \eta_0), \ldots]$ in the element's own coordinates |
| `stress_recovery.calc_von_mises(sigma)` | $\sigma_{\text{VM}} = \sqrt{\sigma_{xx}^2 - \sigma_{xx}\sigma_{yy} + \sigma_{yy}^2 + 3\sigma_{xy}^2}$ for each row of a plane stress array |

## Plotting

`GUI(config, mesh, solver_data)` opens the GUI. "Save Screen" saves the plot as a PNG file in the folder that the
script was run from.

`plot_field(ax, config, mesh, solver_data, field)` draws a field as contours in your own matplotlib axes, and
returns the contours, e.g. for `plt.colorbar(contours)`.

| Problem | Fields (`config.contour` and `plot_field`) |
|------------|------------------------------------------------------------|
| Plane stress | `"none"`, `"u"`, `"v"`, `"u_mag"` (the length of the displacement), `"sigma_xx"`, `"sigma_yy"`, `"sigma_xy"`, `"sigma_vm"` (von Mises) |
| Plate | `"none"`, `"w"`, `"theta_x"`, `"theta_y"`, `"M_xx"`, `"M_yy"`, `"M_xy"`, `"V_x"`, `"V_y"`, `"applied_load"` (the pressure and body force, as a force per area) |

## Short names in the code

| Name | Meaning |
|------------------------|------------------------------------------------------|
| `nN`, `nE` | Number of nodes and elements |
| `nEx`, `nEy` | Number of elements in the $x$- and $y$-direction |
| `nNl` | Number of nodes in an element (l for local) |
| `e`, `il`, `I` | Element number, node number within an element (local), and node number in the mesh (global) |
| `xi`, `eta` | The element's own coordinates $\xi$ and $\eta$, from $-1$ to $1$ |
| `N`, `dNdxi`, `dNdx` | Shape functions and their derivatives |
| `J`, `detJ` | The Jacobian matrix of the mapping from $(\xi, \eta)$ to $(x, y)$, and its determinant |
| `nGauss_1D` | Number of Gauss points in each direction |
| `Ke` | Element stiffness matrix |
| `B`, `Bb`, `Bs` | Strain-displacement matrices (`Bb` and `Bs` for bending and shear in plates) |
| `D`, `D_b`, `D_s` | Material matrices (`D_b` and `D_s` for bending and shear in plates) |
| `r`, `R_ext`, `R_int`, `K` | See [Solver data and results](#solver-data-and-results) |
