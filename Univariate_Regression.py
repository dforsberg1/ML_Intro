import pandas as pd
import math
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 300

# Load the data
data = pd.read_csv('/Users/danielforsberg/Desktop/venv/MachineLearningCourse/ML_05/day.csv')

# Set Seed (So we can get a reproducible set of random integers)
np.random.seed(42)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data['atemp'], data['cnt'], test_size=0.5)

# Convert variables from series to arrays
x_train = x_train.values
y_train = y_train.values
x_test = x_test.values
y_test = y_test.values

# Function to calculate the cost
 
def compute_cost(x, y, b0, b1):

    m = x_train.shape[0] 
    cost = 0

    for i in range(m):
        f_b = b0 + (b1 * x[i])
        cost = cost + (f_b - y[i])**2
    total_cost = 1 / (2 * m) * cost

    return total_cost

# Function to Compute Gradient Descent

def compute_gradient(x, y, b0, b1): 
    
    """
    Computes the gradient for linear regression 
    Args:
      x (ndarray (m,)): Data, m examples 
      y (ndarray (m,)): target values
      b0,b1 (scalars): model parameters  
    Returns
      dj_db0 (scalar): The gradient of the cost w.r.t. the parameters b0
      dj_db1 (scalar): The gradient of the cost w.r.t. the parameter b1  
     """

    # Number of training examples
    m = x.shape[0]    
    dj_db0 = 0
    dj_db1 = 0

    for i in range(m):  
        f_b = b0 + (b1 * x[i])
        dj_db0_i = f_b - y[i] 
        dj_db1_i = (f_b - y[i]) * x[i] 
        dj_db0 += dj_db0_i
        dj_db1 += dj_db1_i 
    dj_db0 = dj_db0 / m
    dj_db1 = dj_db1 / m 
    
    return dj_db0, dj_db1

# Compute Gradient Descent

def gradient_descent(x, y, b0_in, b1_in, alpha, num_iters, cost_function, gradient_function): 
    """
    Performs gradient descent to fit b0,b1. Updates b0,b1 by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      x (ndarray (m,))  : Data, m examples 
      y (ndarray (m,))  : target values
      b0_in, b1_in (scalars): initial values of model parameters  
      alpha (float):     Learning rate
      num_iters (int):   number of iterations to run gradient descent
      cost_function:     function to call to produce cost
      gradient_function: function to call to produce gradient
      
    Returns:
      b0 (scalar): Updated value of intercept parameter after running gradient descent
      b1 (scalar): Updated value of slope parameter after running gradient descent
      J_history (List): History of cost values
      p_history (list): History of parameters [b0,b1] 
      """
   
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    p_history = []
    b0 = b0_in
    b1 = b1_in
    
    for i in range(num_iters):
        # Calculate the gradient and update the parameters using gradient_function
        dj_db0, dj_db1 = gradient_function(x, y, b0, b1)     

        # Update Parameters using equation (3) above
        b0 = b0 - alpha * dj_db0                            
        b1 = b1 - alpha * dj_db1                            

        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append(cost_function(x, y, b0, b1))
            p_history.append([b0,b1])
        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0:
            print(f"Iteration {i:5}: Cost {J_history[-1]:10.1f} ",
                  f"dj_db0: {dj_db0:9.1e}, dj_db1: {dj_db1:9.1e} ",
                  f"b0: {b0:9.2f}, b1:{b1:9.2f}")
 
    return b0, b1, J_history, p_history # return b1 and J, b1 history for graphing

# initialize parameters
b0_in = 0
b1_in = 0

# Some gradient descent settings
iterations = 10000
tmp_alpha = 0.08

# Run gradient descent
b0_final, b1_final, J_hist, p_hist = gradient_descent(x_train, y_train, b0_in, b1_in, tmp_alpha, 
                                                    iterations, compute_cost, compute_gradient)

print(f"(b0,b1) found by gradient descent: ({b0_final:0.2f}, {b1_final:0.2f})")

# Plot the Total Cost as function of b1 while holding b0 = b0_final

# Define the cost function
def cost_function(x, y, b0, b1):
    n = len(x)
    error = (b0 + (b1 * x)) - y
    return np.sum(error ** 2) / (2*n)

# Define range and step size for b1
b1_values = np.arange(b1_final - 15000, b1_final + 15000, 1)

# Calculate costs # this is y = f(x) and is a parabola
costs = [cost_function(x_train, y_train, b0_final, b1) for b1 in b1_values]

# Identify Other Plot Points
b1_points = np.array([(min(b1_values) + b1_final)/2, b1_final, (max(b1_values) + b1_final)/2])
cost_points = [cost_function(x_train, y_train, b0_final, b1) for b1 in b1_points]

def compute_gradient_at_point(x, y, b0, b1): 
    """
    Computes the gradient for linear regression at a particular point on the cost function
    Args:
      x (ndarray (m,)): Data, m examples 
      y (ndarray (m,)): target values
      b0,b1 (scalars): model parameters
      point (scalar): point at which gradient needs to be calculated  
    Returns
      dj_db0 (scalar): The gradient of the cost w.r.t. the parameter b0 at the given point
      dj_db1 (scalar): The gradient of the cost w.r.t. the parameter b1 at the given point
    """
    # Number of training examples
    m = x.shape[0]    
    dj_db0 = 0
    dj_db1 = 0

    for i in range(m):  
        f_b = b0 + (b1 * x[i])
        dj_db0_i = f_b - y[i] 
        dj_db1_i = (f_b - y[i]) * x[i] 
        dj_db0 += dj_db0_i
        dj_db1 += dj_db1_i 
    dj_db0 = dj_db0 / m
    dj_db1 = dj_db1 / m 
    
    if abs(dj_db1) < 1e-5:
        return 0.0
    else:
        return dj_db1

# calculate the gradient at a particular points
b1_gradients = np.round([compute_gradient_at_point(x_train, y_train, b0_final, b1) for b1 in b1_points])
print(f"The minimum cost value = {cost_points[1]:.2f} after optimizing for b0 and b1")

"""
The following graph shows the cost function vs. b1 when b0 is optimized  
"""

# Plot the cost vs. b1
plt.ticklabel_format(style='plain', axis='y')
plt.plot(b1_values, costs)

# Plot and label gradient at specific points
plt.scatter(b1_points, cost_points)
plt.text(b1_points[0] - 800, cost_points[0], r'$\frac{\partial J}{\partial \beta_1}$' + ' = ' + f"{round(b1_gradients[0])}", ha = 'right')
plt.text(b1_points[1], cost_points[1] + 2e6, r'$\frac{\partial J}{\partial \beta_1}$' + ' = ' + f"{round(b1_gradients[1])}", ha = 'center')
plt.text(b1_points[2] + 800, cost_points[2], r'$\frac{\partial J}{\partial \beta_1}$' + ' = ' + f"{round(b1_gradients[2])}", ha = 'left')

# Define the partial derivative w.r.t. b1
def partial_derivative_b1(x, y, b0, b1):
    n = len(x)
    error = (b0 + (b1 * x)) - y
    return np.sum(error * x) / n

# Define tangent line
# y = m*(x - x1) + y1
def tangent_line(x, x1, y1, slope):
    return slope*(x - x1) + y1

# Iterate through each point in b1_points
for b1 in b1_points:
    b1range = np.linspace(b1-3000, b1+3000, 100)
    y1 = cost_function(x_train, y_train, b0_final, b1)  # choose point to plot tangent line
    slope1 = partial_derivative_b1(x_train, y_train, b0_final, b1)  # establish slope parameter
    
    plt.plot(b1range, tangent_line(b1range, b1, y1, slope1), color='darkred', linestyle='dashed', linewidth=2)

# Create labels for x-axis, y-axis, and main title
plt.xlabel(r"$\beta_1$ with $\beta_0$ = " + f"{b0_final:.2f}")
plt.ylabel('Cost')
plt.title(r'Cost vs. $\beta_1$')

plt.show()

"""
Quiver Plot
"""

# Create Quiver Plot

# Compute gradients at each point on the grid
b0, b1 = np.meshgrid(np.linspace(b0_final - 500, b0_final + 500, 15), np.linspace(b1_final - 1000, b1_final + 1000, 15))
grad_b0, grad_b1 = compute_gradient(x_train, y_train, b0, b1)

# Create a quiver plot of the gradients
plt.quiver(b1, b0, grad_b1, grad_b0, color='b')
plt.xlabel(r'$ \beta_1$')
plt.ylabel(r'$ \beta_0$')
plt.title(r'Quiver plot of gradients over values of $\beta_0$ and $\beta_1$')
plt.show()

"""
Plot cost vs. iteration step
"""

# Plot cost versus iteration
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12,5))
ax1.plot(J_hist[:21]) 
ax2.plot(1000 + np.arange(len(J_hist[1000:])), J_hist[1000:])
ax1.set_title("Cost vs. Iteration (start)");  ax2.set_title("Cost vs. Iteration (end)")
ax1.set_ylabel('Cost')            ;  ax2.set_ylabel('Cost') 
ax1.set_xlabel('Iteration step')  ;  ax2.set_xlabel('Iteration step') 

# Set the format of the tick labels on each y-axis
ax1.yaxis.set_major_formatter('{:.0f}'.format)
ax2.yaxis.set_major_formatter('{:.0f}'.format)

plt.show()

"""
Contour plot of cost(b0,b1) over a range of values for b0 and b1
"""

# Define the range of b0 and b1 values
# b0_range = np.linspace(-120, 1879, 100)
# b1_range = np.linspace(6705, 8705, 100)

b0_range = np.linspace(-200, 2000, 100)
b1_range = np.linspace(6700, 8700, 100)

# Create a 2D meshgrid of the b0 and b1 values
b0, b1 = np.meshgrid(b0_range, b1_range)

# Compute the cost for each combination of b0 and b1
cost_vals = np.zeros_like(b0)
for i in range(b0.shape[0]):
    for j in range(b1.shape[0]):
        cost_vals[i,j] = compute_cost(x_train, y_train, b0[i,j], b1[i,j])

# Create a contour plot of the cost function
fig, ax = plt.subplots()
contour = ax.contour(b0, b1, cost_vals, levels=15)
ax.clabel(contour, inline=True, fontsize=8)
ax.set_xlabel(r'$ \beta_0$')
ax.set_ylabel(r'$ \beta_1$')

fig.suptitle(r'Contour Plot of J($\beta$) vs. $\beta_0$,$\beta_1$')
ax.set_title(f'Min(Cost) = {round(cost_points[1])}')
# ax.suptitle(f'Cost minimized at {cost_points[1]}', fontsize=18)             

# Plot the purple dotted lines
ax.plot([ax.get_xlim()[0], b0_final], [b1_final, b1_final], lw=2, color='purple', ls='dotted')
ax.plot([b0_final, b0_final], [ax.get_ylim()[0], b1_final], lw=2, color='purple', ls='dotted')

ax.scatter(x=[b0_final], y=[b1_final], c='purple')      

plt.show()

"""
Using matplotlin: Plot 3D surface plot of cost as a function of parameters b0,b1 w/ path of gradient descent
"""

# Define the b0 and b1 range
b0_range = np.linspace(-1500, 3000, 100)
b1_range = np.linspace(4000, 12000, 100)

# Create a meshgrid of b0 and b1 values
B0, B1 = np.meshgrid(b0_range, b1_range)

# Calculate the total cost for each combination of b0 and b1 using compute_cost function
total_cost = np.zeros_like(B0)

for i in range(B0.shape[0]):
    for j in range(B0.shape[1]):
        total_cost[i, j] = compute_cost(x_train, y_train, B0[i, j], B1[i, j])

# Create a 3D plot of cost function
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(B0, B1, total_cost, cmap='jet', zorder=1)
ax.set_xlabel(r'$ \beta_0$')
ax.set_ylabel(r'$ \beta_1$')
ax.set_zlabel(r'Cost = J($\beta_0$, $\beta_1$)')
ax.set_title(r'3D Contour Plot of J($\beta_0$, $\beta_1$) w/ path of gradient descent')

# Add the path of gradient descent to the plot
J_hist = np.array(J_hist)
p_hist = np.array(p_hist)
ax.plot(p_hist[:, 0], p_hist[:, 1], J_hist, marker='o', color='red', zorder=10)

# Adjust padding to ensure entire plot fits within viewer window
plt.tight_layout()

# Show the plot
plt.show()

"""
Using Plotly: Plot 3D surface plot of cost as a function of parameters b0,b1 w/ path of gradient descent
"""

import plotly.graph_objs as go
import numpy as np
import plotly.io as pio
pio.renderers.default='browser'

# Define the range of values for beta0 and beta1
beta0_vals = np.linspace(-1500, 3000, 100)
beta1_vals = np.linspace(4000, 12000, 100)

# Create a meshgrid of beta0 and beta1 values
beta0_mesh, beta1_mesh = np.meshgrid(beta0_vals, beta1_vals)

# Calculate the cost for each combination of beta0 and beta1
cost_vals = np.zeros((len(beta0_vals), len(beta1_vals)))
for i in range(len(beta0_vals)):
    for j in range(len(beta1_vals)):
        cost_vals[i, j] = compute_cost(x_train, y_train, beta0_vals[i], beta1_vals[j])

# Create the 3D surface plot
fig = go.Figure(data=[go.Surface(x=beta0_mesh, y=beta1_mesh, z=cost_vals)])
fig.update_layout(title='Cost Surface', scene=dict(xaxis_title='x = Beta0', yaxis_title='y = Beta1', zaxis_title='z = Cost'))

# Add the path of gradient descent to the plot
fig.add_trace(go.Scatter3d(x=p_hist[:,0], y=p_hist[:,1], z=J_hist, mode='markers', marker=dict(size=5, color='red')))
fig.show()

pio.renderers.default='svg'

"""
Plot regression line on test data
"""

# Plot the regression line and the training data
plt.scatter(x_train, y_train)
plt.plot(x_train, b0_final + b1_final * x_train, '-r')
plt.xlabel('Normalized temperature feeling [Celsius]')
plt.ylabel('Number of total rentals per day')
plt.title('Fitted Regression Line: Training Data')

plt.show()

"""
Test the model on testing data & plot the prediction line
"""

# Test the Model against Test Data
predictions = b0_final + b1_final * x_test

plt.scatter(x_test, y_test)
plt.plot(x_test, predictions, '-r')
plt.xlabel('Normalized temperature feeling [Celsius]')
plt.ylabel('Number of total rentals per day')
plt.title('Fitted Regression Line: Testing Data')

plt.show()

# Calculate a metric of the model's performance
mse = np.mean((predictions - y_test)**2)
rmse = np.sqrt(mse)
print(f"Model RMSE = {rmse:0.2f}")
# RMSE (or Root Mean Squared Error) measures the average difference between values predicted by a model and the actual values
