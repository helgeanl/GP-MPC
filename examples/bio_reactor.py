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
path.append(r"./../")

import numpy as np
import casadi as ca
import time

from gp_mpc import Model, GP, MPC, plot_eig, lqr


def ode(x, u, z,p=0.5):
    # Model Parameters
    p = 0.5

    dxdt = [
             z[0] * x[0] - x[0] * u[0],
             z[1] * x[1] - x[1] * u[0],
             -p * x[0] * x[2] + u[1] - x[2] * u[0]
            ]
    return ca.vertcat(*dxdt)


def alg(x, z, u):
    # Model Parameters
    mu1_M = 0.4
    mu2_M = 0.5
    K = 0.05
    K_I = 0.02
    y1 = 0.2
    y2 = 0.15
    Sf = 2.0

    alg = [
             mu1_M * (Sf - x[0] / y1 - x[1] / y2) /
                 (K + (Sf - x[0] / y1 - x[1] / y2)) - z[0],
             mu2_M * (Sf - x[0] / y1 - x[1] / y2) * K /
                 ((K + (Sf - x[0] / y1 - x[1] / y2)) * (K_I + x[2])) - z[1]
            ]

    return ca.vertcat(*alg)

def alg_0(x, u):
    # Model Parameters
    mu1_M = 0.4
    mu2_M = 0.5
    K = 0.05
    K_I = 0.02
    y1 = 0.2
    y2 = 0.15
    Sf = 2.0

    alg = [
             mu1_M * (Sf - x[0] / y1 - x[1] / y2) /
                 (K + (Sf - x[0] / y1 - x[1] / y2)),
             mu2_M * (Sf - x[0] / y1 - x[1] / y2) * K /
                 ((K + (Sf - x[0] / y1 - x[1] / y2)) * (K_I + x[2]))
            ]

    return ca.vertcat(*alg)


def inequality_constraints(x, covar, u, eps):
    # Add some constraints here if you need
    con_ineq = []
    con_ineq_lb = []
    con_ineq_ub = []

    cons = dict(con_ineq=con_ineq,
                con_ineq_lb=con_ineq_lb,
                con_ineq_ub=con_ineq_ub
    )
    return cons



solver_opts = {
                'ipopt.linear_solver' : 'ma27',
                'ipopt.max_cpu_time' : 10,
                'expand' : True,
}

meanFunc = 'zero'
dt = .1
Nx = 3
Nu = 2
Nz = 2
R = np.eye(Nx) * 1e-5

# Limits in the training data
ulb = [0., 0.]
uub = [1., .1]
xlb = [.0, .0, .0]
xub = [1., 1., 1.]

N = 10 # Number of training data

# Create simulation model
model          = Model(Nx=Nx, Nu=Nu, Nz=Nz, ode=ode, alg=alg, alg_0=alg_0,
                        dt=dt, R=R, clip_negative=True)
X, Y           = model.generate_training_data(N, uub, ulb, xub, xlb, noise=True)
X_test, Y_test = model.generate_training_data(N, uub, ulb, xub, xlb, noise=True)



# Create GP model
gp = GP(X, Y, normalize=True)
gp.validate(X_test, Y_test)

x0 = np.array([.3, .3, .3])
u0 = np.array([40, 40])
u_test = np.ones((30, 2)) * .3
gp.predict_compare(x0, u_test, model)
model.plot(x0, u_test)

# Linear System
A, B = model.discrete_linearize(x0, u0)
plot_eig(A)

## Limits in the MPC problem
ulb = [0., 0.]
uub = [1., .1]
xlb = [.0, .0, .0]
xub = [1., 1., 1.]
x_sp = np.array([.5, .5, .5])

#mpc = MPC(horizon=12*dt, gp=gp, model=model,
#          gp_method='ME',
#          ulb=ulb, uub=uub, xlb=xlb, xub=xub,
#          terminal_constraint=0, costFunc='quad', feedback=False,
#          solver_opts=solver_opts, discrete_method='rk4',
#          inequality_constraints=None
#          )
#
#
#x, u = mpc.solve(x0, sim_time=15*dt, x_sp=x_sp, debug=False)
#mpc.plot()
