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
    x_min, x_max = geom.xmin[0], geom.xmax[0] #Retrieves the left and right x-coordinates of the domain.

    def left_boundary(x, on_boundary):
        return on_boundary and np.isclose(x[0], x_min) #The left boundary is fixed (both displacements, u and v, are zero)

    def right_boundary(x, on_boundary):
        return on_boundary and np.isclose(x[0], x_max) #The right boundary has a prescribed displacement in the x-direction (0.1) and zero in the y-direction.

    bc_left = dde.DirichletBC(geom, lambda x: [0, 0], left_boundary)
    bc_right = dde.DirichletBC(geom, lambda x: [0.1, 0], right_boundary)
    return [bc_left, bc_right] #Returns a list of boundary conditions


def train_pinn_with_geometry(bbox, num_epochs=5000):
    """
    Train a PINN model using DeepXDE to solve the 2D linear elasticity PDE
    on a bounding box derived from the segmentation.
    bbox: ([x_min, y_min], [x_max, y_max])
    """
    # Define geometry from bounding box
    geom = dde.geometry.Rectangle(bbox[0], bbox[1]) #A rectangle is created using the bounding box. This domain represents the region over which the PDE will be solved

    # Define the PDE Function: Wraps the elasticity_pde function so that DeepXDE can call it during training.
    def pde(x, y):
        res = elasticity_pde(x, y)
        return [res[0], res[1]]

    bcs = boundary_conditions(geom)
    # Set Up the Data Object: num_domain-Number of collocation points inside the domain, num_boundary - Number of points on the boundary.
    data = dde.data.TimePDE(geom, pde, bcs, num_domain=1024, num_boundary=50)

    # Neural Network, Input dimension: 2 (for x and y), Hidden layers: 4 layers of 50 neurons each, using the hyperbolic tangent activation function ("tanh")., 
    net = dde.maps.FNN([2] + [50]*4 + [2], "tanh", "Glorot uniform") # Output dimension: 2 (displacement components u and v), "Glorot uniform" initialization is used to start with reasonable weights.


    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3)
    model.train(epochs=num_epochs)
    model.compile("L-BFGS")
    model.train()
    return model, model.predict #Returns the trained model and its prediction function (model.predict), which can be used to compute the displacement field on any set of input point
