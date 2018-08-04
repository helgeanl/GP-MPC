# -*- coding: utf-8 -*-
"""
@author: Helge-André Langåker
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sys import path
path.append(r"./../") # Add gp_mpc pagkage to path

import numpy as np
import casadi as ca
from gp_mpc import Model, GP, MPC


def ode(x, u, z, p):
    # Model Parameters (Raff, Tobias et al., 2006)
    g = 981
    a1 = 0.233
    a2 = 0.242
    a3 = 0.127
    a4 = 0.127
    A1 = 50.27
    A2 = 50.27
    A3 = 28.27
    A4 = 28.27
    gamma1 = 0.4
    gamma2 = 0.4

    dxdt = [
            (-a1 / A1) * ca.sqrt(2 * g * x[0] + 1e-3) + (a3 / A1 )
                * ca.sqrt(2 * g * x[2] + 1e-3) + (gamma1 / A1) * u[0],
            (-a2 / A2) * ca.sqrt(2 * g * x[1] + 1e-3) + a4 / A2
                * ca.sqrt(2 * g * x[3]+ 1e-3) + (gamma2 / A2) * u[1],
            (-a3 / A3) * ca.sqrt(2 * g * x[2] + 1e-3) + (1 - gamma2) / A3 * u[1],
                (-a4 / A4) * ca.sqrt(2 * g * x[3] + 1e-3) + (1 - gamma1) / A4 * u[0]
    ]

    return  ca.vertcat(*dxdt)


""" System parameters """
dt = 3.0
Nx = 4
Nu = 2
R = np.eye(Nx) * 1e-5 # Noise covariance

""" Limits in the training data """
ulb = [0., 0.]
uub = [60., 60.]
xlb = [.0, .0, .0, .0]
xub = [30., 30., 30., 30.]

N = 60          # Number of training data
N_test = 100    # Number of test data

""" Create Simulation Model """
model          = Model(Nx=Nx, Nu=Nu, ode=ode, dt=dt, R=R, clip_negative=True)
X, Y           = model.generate_training_data(N, uub, ulb, xub, xlb, noise=True)
X_test, Y_test = model.generate_training_data(N_test, uub, ulb, xub, xlb, noise=True)


if 1:
    """ Create GP model and optimize hyper-parameters on training data """
    gp = GP(X, Y, mean_func='zero', normalize=True, xlb=xlb, xub=xub, ulb=ulb,
            uub=uub)
    gp.save_model('models/gp_tank')
else:
    """ Or Load Example Model"""
    gp = GP.load_model('models/gp_tank_example')
gp.validate(X_test, Y_test)
gp.print_hyper_parameters()

""" Limits in the MPC problem """
ulb = [10., 10.]
uub = [60., 60.]
xlb = [7.5, 7.5, 3.5, 4.5]
xub = [28., 28., 28., 28.]

""" Initial state, input and set point  """
x_sp = np.array([14.0, 14.0, 14.2, 21.3])
x0 = np.array([8., 10., 8., 19.])
u0 = np.array([45, 45])

""" Penalty matrices """
Q = np.diag([20, 20, 10, 10])   # State penalty
R = np.diag([1e-3, 1e-3])       # Input penalty
S = np.diag([.01, .01])         # Input change penalty

""" Options to pass to the MPC solver """
solver_opts = {
        #    'ipopt.linear_solver' : 'ma27',    # Plugin solver from HSL
            'ipopt.max_cpu_time' : 30,
            'expand' : True,
}

""" Build MPC solver """
mpc = MPC(horizon=30*dt, gp=gp, model=model,
           gp_method='TA',
           ulb=ulb, uub=uub, xlb=xlb, xub=xub, Q=Q, R=R, S=S,
           terminal_constraint=None, costFunc='quad', feedback=True,
           solver_opts=solver_opts, discrete_method='gp',
           inequality_constraints=None
           )

""" Solve and plot the MPC solution, simulating 80 iterations """
x, u = mpc.solve(x0, u0=u0, sim_time=80*dt, x_sp=x_sp, debug=False, noise=True)
mpc.plot(xnames=['Tank 1 [cm]', 'Tank 2 [cm]','Tank 3 [cm]','Tank 4 [cm]'],
        unames=['Pump 1 [ml/s]', 'Pump 2 [ml/s]'])
