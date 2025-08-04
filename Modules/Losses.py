import torch
import numpy as np

"""
Here is some naming
output = net approximation where data is
true_y = collected data points

fg = where we evaluate the net outside of the data
fg_y = neural net where there is no data
"""

def d_loss(output, true_y, fg, fg_y, u, h):

    mse_loss = torch.nn.MSELoss()(output, true_y)

    grad_outputs = torch.ones_like(fg_y)
    fg_y = fg_y.requires_grad_(True)
    gradients = torch.autograd.grad(outputs = fg_y,
                               inputs = fg,
                               grad_outputs = grad_outputs,
                               create_graph = True)[0]

    derivative_x1 = gradients[:, 0]
    derivative_x2 = gradients[:, 1]

    derivative_term =derivative_x1.pow(2).mean() + h*derivative_x2.pow(2).mean()
    
    return mse_loss + u*derivative_term


# def sp_t_reg(batch_x,net,lx,ly,lt):
#     x = (batch_x[:,0].reshape(-1,1)).requires_grad_(True)
#     y = (batch_x[:,1].reshape(-1,1)).requires_grad_(True)
#     t = (batch_x[:,2].reshape(-1,1)).requires_grad_(True)
#     outputs = net(batch_x).requires_grad_(True)

#     grad_outputs = torch.ones_like(outputs)
#     gradients = torch.autograd.grad(outputs= outputs, 
#                                     inputs= [x,y,t],
#                                     grad_outputs=grad_outputs,
#                                     create_graph=True,
#                                     retain_graph=True,
#                                     allow_unused=True)
    
#     grad_x, grad_y, grad_t = gradients

#     space = lx*torch.mean(grad_x**2) +ly*torch.mean(grad_y**2)
#     # time = lt*torch.mean(grad_t**2)
#     sp_t_loss = space
    
#     return sp_t_loss

def sp_t_reg(batch_x, net, lx, ly, lt):
    """
    Fixed version with proper tensor handling
    """
    print(f"Input batch_x shape: {batch_x.shape}")
    
    # Ensure we're working with the right dimensions
    if batch_x.dim() == 3:  # If batch_x is [1, n_points, 3]
        batch_x = batch_x.squeeze(0)  # Make it [n_points, 3]
    
    n_points = batch_x.shape[0]
    print(f"Processing {n_points} spatial points")
    
    # Create input coordinates that require gradients - keep as 2D
    x = batch_x[:, 0:1].clone().detach().requires_grad_(True)  # [n_points, 1]
    y = batch_x[:, 1:2].clone().detach().requires_grad_(True)  # [n_points, 1]
    t = batch_x[:, 2:3].clone().detach().requires_grad_(True)  # [n_points, 1]
    
    # Combine into input tensor
    coords = torch.cat([x, y, t], dim=1)  # [n_points, 3]
    print(f"Coords shape: {coords.shape}")
    
    # Get network predictions
    predictions = net(coords)
    print(f"Predictions shape: {predictions.shape}")
    
    # For gradient computation, we need a scalar output
    scalar_output = predictions.sum()
    
    # Compute gradients
    grad_x = torch.autograd.grad(scalar_output, x, create_graph=True, retain_graph=True)[0]
    grad_y = torch.autograd.grad(scalar_output, y, create_graph=True, retain_graph=True)[0]
    grad_t = torch.autograd.grad(scalar_output, t, create_graph=True, retain_graph=True)[0]
    
    print(f"grad_x shape: {grad_x.shape}, mean_squared: {torch.mean(grad_x**2):.6f}")
    print(f"grad_y shape: {grad_y.shape}, mean_squared: {torch.mean(grad_y**2):.6f}")
    print(f"grad_t shape: {grad_t.shape}, mean_squared: {torch.mean(grad_t**2):.6f}")
    
    # Compute spatial regularization
    x_contrib = lx * torch.mean(grad_x**2)
    y_contrib = ly * torch.mean(grad_y**2)
    
    total_spatial_reg = x_contrib + y_contrib
    print(f"Spatial regularization: {total_spatial_reg:.6f}")
    
    return total_spatial_reg


def crit( MSE = False, dloss = False, sp_t = False, params = None, net =None):
    mse = torch.nn.MSELoss()
    if MSE:
        def cust(output, true_y, *args):
            return mse(output,true_y)
        return cust
    elif dloss:
        u, h = params
        def close(output, true_y, fg, fg_y):
            return d_loss(output, true_y, fg, fg_y,u,h)
        return close
    elif sp_t:
        lx, ly, lt = params
        def closed(batch_x, batch_y):
            return sp_t_reg(batch_x,net, lx, ly, lt) + mse(net(batch_x), batch_y)
        return closed

"""Takes the objects made by covariance in 
Cov."""

# def c_loss(U,L, predictions_on_grid ,predicted_y, true_y, u):
#     mse = torch.nn.MSELoss()(predicted_y, true_y)

#     cov = L**(-1/2) (U.T @ predictions_on_grid)  # we should be left with a small vector
#     cov_loss = torch.norm(cov, p=2)
#     return mse + u*cov_loss

"""This is the RMS used for comparason"""

def rms(data,y_approx,std):
    return torch.sqrt(torch.sum(((data-y_approx)/std)**2)/len(data))

def continuity_loss(prior,current):
    diff = current.reshape(-1,1)- prior.detach().numpy().reshape(-1, 1)  # Reshape prior to match current's shape
    # Ensure diff is a numpy array for mean calculation
    if not isinstance(diff, np.ndarray):
        diff = np.array(diff)
    return torch.tensor(np.sqrt(np.mean(diff**2)), dtype=torch.float32, requires_grad= True)  # Mean squared error between prior and current prediction


# def sp_t_reg(batch_x, net, lx, ly, lt):
#     """
#     Fixed version with proper tensor handling
#     """
#     print(f"Input batch_x shape: {batch_x.shape}")
    
#     # Ensure we're working with the right dimensions
#     if batch_x.dim() == 3:  # If batch_x is [1, n_points, 3]
#         batch_x = batch_x.squeeze(0)  # Make it [n_points, 3]
    
#     n_points = batch_x.shape[0]
#     print(f"Processing {n_points} spatial points")
    
#     # Create input coordinates that require gradients - keep as 2D
#     x = batch_x[:, 0:1].clone().detach().requires_grad_(True)  # [n_points, 1]
#     y = batch_x[:, 1:2].clone().detach().requires_grad_(True)  # [n_points, 1]
#     t = batch_x[:, 2:3].clone().detach().requires_grad_(True)  # [n_points, 1]
    
#     # Combine into input tensor
#     coords = torch.cat([x, y, t], dim=1)  # [n_points, 3]
#     print(f"Coords shape: {coords.shape}")
    
#     # Get network predictions
#     predictions = net(coords)
#     print(f"Predictions shape: {predictions.shape}")
    
#     # For gradient computation, we need a scalar output
#     scalar_output = predictions.sum()
    
#     # Compute gradients
#     grad_x = torch.autograd.grad(scalar_output, x, create_graph=True, retain_graph=True)[0]
#     grad_y = torch.autograd.grad(scalar_output, y, create_graph=True, retain_graph=True)[0]
#     grad_t = torch.autograd.grad(scalar_output, t, create_graph=True, retain_graph=True)[0]
    
#     print(f"grad_x shape: {grad_x.shape}, mean_squared: {torch.mean(grad_x**2):.6f}")
#     print(f"grad_y shape: {grad_y.shape}, mean_squared: {torch.mean(grad_y**2):.6f}")
#     print(f"grad_t shape: {grad_t.shape}, mean_squared: {torch.mean(grad_t**2):.6f}")
    
#     # Compute spatial regularization
#     x_contrib = lx * torch.mean(grad_x**2)
#     y_contrib = ly * torch.mean(grad_y**2)
    
#     total_spatial_reg = x_contrib + y_contrib
#     print(f"Spatial regularization: {total_spatial_reg:.6f}")
    
#     return total_spatial_reg