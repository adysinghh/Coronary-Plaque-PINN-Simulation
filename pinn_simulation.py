# #!/usr/bin/env python
# """
# Physics-Informed Neural Network (PINN) module using DeepXDE.
# This module defines the PDE for 2D linear elasticity (as a proxy for stent expansion),
# sets boundary conditions, and trains a PINN model.
# """

# import numpy as np
# import deepxde as dde

# def elasticity_pde(x, y):
#     """
#     Defines the PDE for 2D linear elasticity.
#     u = [u, v] is the displacement field.
#     PDE: -div(sigma) = 0, where sigma = λ*(div(u))*I + 2μ*ε(u)
#     """
#     lam = 1.0  # First Lamé constant
#     mu = 1.0   # Second Lamé constant

#     u = y[:, 0:1]
#     v = y[:, 1:2]

#     du_dx = dde.grad.jacobian(y, x, i=0, j=0)
#     du_dy = dde.grad.jacobian(y, x, i=0, j=1)
#     dv_dx = dde.grad.jacobian(y, x, i=1, j=0)
#     dv_dy = dde.grad.jacobian(y, x, i=1, j=1)

#     div_u = du_dx + dv_dy
#     eps_xx = du_dx
#     eps_yy = dv_dy
#     eps_xy = 0.5 * (du_dy + dv_dx)

#     sigma_xx = lam * div_u + 2 * mu * eps_xx
#     sigma_yy = lam * div_u + 2 * mu * eps_yy
#     sigma_xy = 2 * mu * eps_xy

#     f1 = dde.grad.jacobian(sigma_xx, x, i=0, j=0) + dde.grad.jacobian(sigma_xy, x, i=0, j=1)
#     f2 = dde.grad.jacobian(sigma_xy, x, i=0, j=0) + dde.grad.jacobian(sigma_yy, x, i=0, j=1)
#     return [f1, f2]

# def boundary_conditions(geom):
#     """
#     Define Dirichlet boundary conditions for the elasticity problem.
#     Fix u=0 on left boundary and prescribe a displacement on right boundary.
#     """
#     def left_boundary(x, on_boundary):
#         return on_boundary and np.isclose(x[0], 0)

#     def right_boundary(x, on_boundary):
#         return on_boundary and np.isclose(x[0], 1)

#     bc_left = dde.DirichletBC(geom, lambda x: [0, 0], left_boundary)
#     bc_right = dde.DirichletBC(geom, lambda x: [0.1, 0], right_boundary)
#     return [bc_left, bc_right]

# def train_pinn(num_epochs=5000):
#     """
#     Train a PINN model using DeepXDE to solve the 2D linear elasticity PDE.
#     Returns the trained model and the prediction function.
#     """
#     geom = dde.geometry.Rectangle([0, 0], [1, 1])
#     def pde(x, y):
#         res = elasticity_pde(x, y)
#         return [res[0], res[1]]
#     bcs = boundary_conditions(geom)
#     data = dde.data.TimePDE(geom, pde, bcs, num_domain=2560, num_boundary=80)
#     net = dde.maps.FNN([2] + [50]*4 + [2], "tanh", "Glorot uniform")
#     model = dde.Model(data, net)
#     model.compile("adam", lr=1e-3)
#     model.train(epochs=num_epochs)
#     model.compile("L-BFGS")
#     model.train()
#     return model, model.predict
#!/usr/bin/env python
"""
Physics-Informed Neural Network (PINN) module using DeepXDE.
This module defines the PDE for 2D linear elasticity (as a proxy for stent expansion),
sets boundary conditions, and trains a PINN model on the bounding box from the segmentation.
"""

import numpy as np
import deepxde as dde

def elasticity_pde(x, y):
    """
    Defines the PDE for 2D linear elasticity.
    PDE: -div(sigma) = 0, where sigma = lam*(div(u))*I + 2*mu*epsilon(u).
    """
    lam = 1.0  # Lamé constant λ
    mu = 1.0   # Lamé constant μ

    u = y[:, 0:1]
    v = y[:, 1:2]

    du_dx = dde.grad.jacobian(y, x, i=0, j=0)
    du_dy = dde.grad.jacobian(y, x, i=0, j=1)
    dv_dx = dde.grad.jacobian(y, x, i=1, j=0)
    dv_dy = dde.grad.jacobian(y, x, i=1, j=1)

    div_u = du_dx + dv_dy
    eps_xx = du_dx
    eps_yy = dv_dy
    eps_xy = 0.5 * (du_dy + dv_dx)

    sigma_xx = lam * div_u + 2 * mu * eps_xx
    sigma_yy = lam * div_u + 2 * mu * eps_yy
    sigma_xy = 2 * mu * eps_xy

    f1 = dde.grad.jacobian(sigma_xx, x, i=0, j=0) + dde.grad.jacobian(sigma_xy, x, i=0, j=1)
    f2 = dde.grad.jacobian(sigma_xy, x, i=0, j=0) + dde.grad.jacobian(sigma_yy, x, i=0, j=1)
    return [f1, f2]

def boundary_conditions(geom):
    """
    Define Dirichlet boundary conditions for the bounding box geometry.
    We fix displacement = 0 at the left boundary and prescribe a displacement on the right boundary.
    """
    # Use geom.xmin and geom.xmax which are lists representing the lower and upper bounds
    x_min, x_max = geom.xmin[0], geom.xmax[0]

    def left_boundary(x, on_boundary):
        return on_boundary and np.isclose(x[0], x_min)

    def right_boundary(x, on_boundary):
        return on_boundary and np.isclose(x[0], x_max)

    bc_left = dde.DirichletBC(geom, lambda x: [0, 0], left_boundary)
    bc_right = dde.DirichletBC(geom, lambda x: [0.1, 0], right_boundary)
    return [bc_left, bc_right]


def train_pinn_with_geometry(bbox, num_epochs=5000):
    """
    Train a PINN model using DeepXDE to solve the 2D linear elasticity PDE
    on a bounding box derived from the segmentation.
    bbox: ([x_min, y_min], [x_max, y_max])
    """
    # Define geometry from bounding box
    geom = dde.geometry.Rectangle(bbox[0], bbox[1])

    def pde(x, y):
        res = elasticity_pde(x, y)
        return [res[0], res[1]]

    bcs = boundary_conditions(geom)
    data = dde.data.TimePDE(geom, pde, bcs, num_domain=1024, num_boundary=50)
    net = dde.maps.FNN([2] + [50]*4 + [2], "tanh", "Glorot uniform")
    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3)
    model.train(epochs=num_epochs)
    model.compile("L-BFGS")
    model.train()
    return model, model.predict
