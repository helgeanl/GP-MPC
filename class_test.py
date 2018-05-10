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
import time
from gp_casadi import GP


from gp_casadi.mpc_class import MPC
from simulation.model import Model


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
    return np.array(dxdt)




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
xub = [25, .5, 2.0, 2.0, 10, 1]

N = 40 # Number of training data
model = Model(Nx=Nx, Nu=Nu, ode=ode, dt=dt, R=R)
X, Y = model.generate_training_data(N, uub, ulb, xub, xlb, noise=True)
X_test, Y_test = model.generate_training_data(N, uub, ulb, xub, xlb, noise=True)

gp = GP(X, Y)
gp.validate(X_test, Y_test)

x0 = np.array([10, 0.0, 0.0, 0.0, 0.0 , 0.0])
u_test = np.zeros((30, 3))
gp.predict_compare(x0, u_test, model)

# Limits in the MPC problem
ulb = [-.5, -.5, -.1,]
uub = [.5, .5, .1,]
xlb = [10, -.5, -2.0, -2.0, .0, .0]
xub = [20, .5, 2.0, 2.0, np.inf, np.inf]

