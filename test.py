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
center = np.argmin((mesh.nodes[:, 0] - L / 2)**2 + (mesh.nodes[:, 1] - L / 2)**2)  #The node closest to the center
D = E * h**3 / (12 * (1 - nu**2))
print(f"Center deflection: {w[center]:.4e}. Kirchhoff theory: {-0.00406 * p * L**4 / D:.4e}")

config.contour = "M_xx"
Plot(config, mesh, solver_data)
