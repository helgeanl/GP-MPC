# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 19:12:26 2018

@author: Helge-André Langåker
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sys import path
path.append(r"C:\Users\helgeanl\Google Drive\NTNU\Masteroppgave\casadi-py36-v3.4.0")
#path.append(r"C:\Users\helgeanl\Google Drive\NTNU\Masteroppgave\casadi-py36-v3.4.1-64bit")
path.append(r"./GP_MPC/")

import numpy as np
import casadi as ca

from gp_mpc import Model, GP, MPC


def ode(x, u):
    # Model Parameters (Gao et al., 2014)
    g   = 9.18                          # Gravity [m/s^2]
    m   = 2050                          # Vehicle mass [kg]
    Iz  = 3344                          # Yaw inertia [kg*m^2]
    Cr   = 65000                        # Tyre corning stiffness [N/rad]
    Cf   = 65000                        # Tyre corning stiffness [N/rad]
    mu  = 0.5                           # Tyre friction coefficient
    l   = 4.0                           # Vehicle length
    lf  = 2.0                           # Distance from CG to the front tyre
    lr  = l - lf                        # Distance from CG to the rear tyre
    Fzf = lr * m * g / (2 * l)          # Vertical load on front wheels
    Fzr = lf * m * g / (2 * l)          # Vertical load on rear wheels
    eps = 1e-8                          # Small epsilon to avoid dividing by zero

    dxdt = [
                1/m * (m*x[1]*x[2] + 2*mu*Fzf*u[0] + 2*Cf*u[2]**2
                    - 2*Cf*u[2] * (x[1] + lf*x[2]) / (x[0] + eps) + 2*mu*Fzr*u[1]),
                1/m * (-m*x[0]*x[2] + 2*mu*Fzf*u[2]*u[0]
                    + 2*Cf*(x[1] + lf*x[2]) / (x[0] + eps) - 2*Cf*u[2]
                    + 2*Cr*(x[1] - lf*x[2]) / (x[0] + eps)),
                1/Iz * (2*lf*mu*Fzf*u[0]*u[2] + 2*lf*Cf*(x[1] + lf*x[2]) / (x[0] + eps)
                    - 2*lf*Cf*u[2] - 2*lr*Cr*(x[1] - lf*x[2]) / (x[0] + eps)),
                x[2],
                x[0]*ca.cos(x[3]) - x[1]*ca.sin(x[3]),
                x[0]*ca.sin(x[3]) + x[1]*ca.cos(x[3])
            ]
    return  ca.vertcat(*dxdt)


def inequality_constraints(x, covar, u, eps):
    # Slip angle constraint
    dx_s = ca.SX.sym('dx')
    dy_s = ca.SX.sym('dy')
    dpsi_s = ca.SX.sym('dpsi')
    delta_f_s = ca.SX.sym('delta_f') 
    lf  = 2.0 
    lr  = 2.0
    slip_min = -4 * np.pi / 180
    slip_max = 4 * np.pi / 180
    slip_f = ca.Function('slip_f', [dx_s, dy_s, dpsi_s, delta_f_s],
                         [(dy_s + lf*dpsi_s)/(dx_s + 1e-6)   - delta_f_s])
    slip_r = ca.Function('slip_r', [dx_s, dy_s, dpsi_s],
                         [(dy_s - lr*dpsi_s)/(dx_s + 1e-6)])
    con_ineq = []
    con_ineq_lb = []
    con_ineq_ub = []
    
    con_ineq.append(slip_f(x[0], x[1], x[2], u[2]) - slip_max - eps)
    con_ineq_ub.append(0)
    con_ineq_lb.append(-np.inf)
    
    con_ineq.append(slip_min - slip_f(x[0], x[1], x[2], u[2]) - eps)
    con_ineq_ub.append(0)
    con_ineq_lb.append(-np.inf)
    
    con_ineq.append(slip_r(x[0], x[1], x[2]) - slip_max - eps)
    con_ineq_ub.append(0)
    con_ineq_lb.append(-np.inf)
    
    con_ineq.append(slip_min - slip_r(x[0], x[1], x[2]) - eps)
    con_ineq_ub.append(0)
    con_ineq_lb.append(-np.inf)
    cons = dict(con_ineq=con_ineq,
                con_ineq_lb=con_ineq_lb,
                con_ineq_ub=con_ineq_ub
            )
    return cons



solver_opts = {}
solver_opts['ipopt.linear_solver'] = 'ma27'
solver_opts['ipopt.max_cpu_time'] = 10
solver_opts['expand']= False    

meanFunc = 'zero'
dt = 0.03
Nx = 6
Nu = 3
R = np.eye(Nx) * 1e-5 

# Limits in the training data
ulb = [-.5, -.5, -.1,]
uub = [.5, .5, .1,]
xlb = [5.0, -.5, -2.0, -2.0, .0, .0]
xub = [25.0, .5, 2.0, 2.0, 10, 1]

N = 20 # Number of training data

# Create simulation model
model          = Model(Nx=Nx, Nu=Nu, ode=ode, dt=dt, R=R)
X, Y           = model.generate_training_data(N, uub, ulb, xub, xlb, noise=True)
X_test, Y_test = model.generate_training_data(N, uub, ulb, xub, xlb, noise=True)

# Create GP model
gp = GP(X, Y)
gp.validate(X_test, Y_test)

x0 = np.array([10, 0.0, 0.0, 0.0, 0.0 , 0.0])
u_test = np.zeros((30, 3))
gp.predict_compare(x0, u_test, model)

# Limits in the MPC problem
ulb = [-.5, -.5, -.1,]
uub = [.5, .5, .1,]
xlb = [5.0, -.5, -2.0, -2.0, .0, .0]
xub = [25.0, .5, 2.0, 2.0, np.inf, np.inf]
x_sp = np.array([10, 0., 0., 0., 20., 0. ])

P = np.array([[0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, 0]])
Q = np.array([[0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 0]])

mpc = MPC(horizon=1*dt, gp=gp, model=model,
          gp_method='EM',
          ulb=ulb, uub=uub, xlb=xlb, xub=xub, Q=Q, P=P,
          terminal_constraint=None, costFunc='quad', feedback=False, 
          solver_opts=solver_opts, discrete_model=True,
          inequality_constraints=inequality_constraints
          )


x, u = mpc.solve(x0, sim_time=4*dt, x_sp=x_sp)
mpc.plot()

