#-----------------------------------------------

# This Code is Please Run in Jupyter Notebook.

#-----------------------------------------------

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
%matplotlib inline

#-----------------------------------------------

def func_lorenz(t, y):
    p = 10.
    r = 28.
    b = 8./3.
    
    dy0dt = -p*y[0] + p*y[1]
    dy1dt = -y[0]*y[2] + r*y[0] - y[1]
    dy2dt = y[0]*y[1] - b*y[2]
    
    return np.stack([dy0dt, dy1dt, dy2dt], axis=0)

#-----------------------------------------------

X, Y, Z = np.meshgrid(np.linspace(-30, 30, 5), np.linspace(-30, 30, 5), np.linspace(0, 60, 5))

X, Y, Z = X.flatten(), Y.flatten(), Z.flatten()

dydt = func_lorenz(None, [X, Y, Z])

fig = plt.figure(figsize=(8, 8))
ax = Axes3D(fig)
ax.quiver(X, Y, Z, dydt[0], dydt[1], dydt[2], length=0.01)
ax.set_xlim(-30, 30)
ax.set_xlabel('y0')
ax.set_ylim(-30, 30)
ax.set_ylabel('y1')
ax.set_zlim(0, 60)
ax.set_zlabel('y2')
ax.set_aspect('equal')

#-----------------------------------------------

def runge_kutta(func, t, y, h):
    k1 = h*func(t, y)
    k2 = h*func(t + h/2, y + k1/2)
    k3 = h*func(t + h/2, y + k2/2)
    k4 = h*func(t + h, y + k3)
    
    return y + k1/6 + k2/3 + k3/3 + k4/6

#-----------------------------------------------

y0_list = [np.array([-30., -30.,    0.]),
                  np.array([ 30., -30.,    0.]),
                  np.array([-30.,  30.,    0.]),
                  np.array([ 30.,  30.,    0.]),
                  np.array([-30., -30., 60.]),
                  np.array([ 30., -30., 60.]),
                  np.array([-30.,  30., 60.]),
                  np.array([ 30.,  30., 60.])]

t_start, t_end = 0., 2.

h = 0.01

t_eval = np.arange(t_start, t_end, h)

y_list = []

for y0 in y0_list:
    y = [y0]
    for t in t_eval[:-1]:
        y_next = runge_kutta(func_lorenz, t, y[-1], h)
        y.append(y_next)
        
    y = np.stack(y, axis=1)
    y_list.append(y)

#-----------------------------------------------

fig = plt.figure(figsize=(8, 8))
ax = Axes3D(fig)
for y in y_list:
    ax.scatter(y[0], y[1], y[2], marker='.')
    
ax.set_xlabel('y0')
ax.set_xlim(-30, 30)
ax.set_ylabel('y1')
ax.set_ylim(-30, 30)
ax.set_zlabel('y2')
ax.set_zlim(0, 60)
ax.set_aspect('equal')

#-----------------------------------------------

fig = plt.figure(figsize=(8, 8))
ax = Axes3D(fig)
cmap = plt.get_cmap('tab10')

for i, y in enumerate(y_list):
    dydt = func_lorenz(None, y)
    ax.quiver(y[0], y[1], y[2], dydt[0], dydt[1], dydt[2], colors=cmap(i), length=0.01)
    
ax.set_xlabel('y0')
ax.set_xlim(-30, 30)
ax.set_ylabel('y1')
ax.set_ylim(-30, 30)
ax.set_zlabel('y2')
ax.set_zlim(0, 60)
ax.set_aspect('equal')

#-----------------------------------------------

from scipy.integrate import solve_ivp

sol_list = []
for y0 in y0_list:
    sol = solve_ivp(func_lorenz, [t_start, t_end], y0, vectorized = True, dense_output=True)
    sol_list.append(sol)
    
dense_eval = np.arange(t_start, t_end, h*0.1)

y_list = []

for sol in sol_list:
    y_list.append(sol.sol(dense_eval))
    
fig = plt.figure(figsize=(8, 8))
ax = Axes3D(fig)
for y in y_list:
    ax.scatter(y[0], y[1], y[2], marker='.')
    
ax.set_xlabel('y0')
ax.set_xlim(-30, 30)
ax.set_ylabel('y1')
ax.set_ylim(-30, 30)
ax.set_zlabel('y2')
ax.set_zlim(0, 60)
ax.set_aspect('equal')

#-----------------------------------------------

