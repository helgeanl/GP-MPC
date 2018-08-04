# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 19:12:26 2018

@author: Helge-André Langåker
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sys import path
path.append(r"./../")

import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from gp_mpc import Model, GP, MPC, plot_eig, lqr

def plot_feedback():
    Nt = 20
    x0 = np.array([8., 10., 8., 18.])
    u0 = np.array([45, 34])
    cov = np.zeros((6,6))
    x = np.zeros((Nt,2))
    x_sim = np.zeros((Nt,2))
    #x_rk4 = np.zeros((Nt,2))

    x[0] = x0
    x_sim[0] = x0
    #x_rk4[0] = x0
    for i in range(Nt-1):
        z = ca
        x_t, cov_x = gp.predict(x[i], [], cov)
        cov[:4,:4] = cov_x
        x[i + 1] = np.array(x_t).flatten()
        x_sim[i+1] = model.integrate(x0=x_sim[i], u=[], p=[])
    #    x_rk4[i+1] = np.array(model.rk4(x_rk4[i], [],[])).flatten()

    plt.figure()
    ax = plt.subplot(111)
    ax.plot(x_sim[:,0], x_sim[:,1], 'k-', linewidth=1.0, label='Exact')
    ax.plot(x[:,0], x[:,1], 'b-', linewidth=1.0, label='GP')
    #ax.plot(x_rk4[:,0], x_rk4[:,1], 'g--', linewidth=1.0, label='RK4')
    ax.set_ylabel('y')
    ax.set_xlabel('x')
    plt.legend(loc='best')
    plt.show()



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





meanFunc = 'zero'
dt = 3.0
Nx = 4
Nu = 2
R = np.eye(Nx) * 1e-5

# Limits in the training data
ulb = [0., 0.]
uub = [60., 60.]
xlb = [.0, .0, .0, .0]
xub = [30., 30., 30., 30.]

N = 60 # Number of training data

# Create simulation model
model          = Model(Nx=Nx, Nu=Nu, ode=ode, dt=dt, R=R, clip_negative=True)
X, Y           = model.generate_training_data(N, uub, ulb, xub, xlb, noise=True)
X_test, Y_test = model.generate_training_data(100, uub, ulb, xub, xlb, noise=True)

# Create GP model
#gp = GP(X, Y, mean_func=meanFunc, normalize=True, xlb=xlb, xub=xub, ulb=ulb,
#        uub=uub, optimizer_opts=None, multistart=1)
#gp.save_model('gp_tank_final2')
gp = GP.load_model('gp_tank_final2')
gp.validate(X_test, Y_test)

gp.print_hyper_parameters()

# Limits in the MPC problem
ulb = [10., 10.]
uub = [60., 60.]
xlb = [7.5, 7.5, 3.5, 4.5]
xub = [28., 28., 28., 28.]
x_sp = np.array([14.0, 14.0, 14.2, 21.3])
x0 = np.array([8., 10., 8., 19.])
u0 = np.array([45, 45])
u_test = np.full((30, 2), [45, 45])

Q = np.diag([20, 20, 10, 10])
#Q = np.diag([10, 10, 1, 1])
R = np.diag([1e-3, 1e-3])
S = np.diag([.01, .01])
#model.check_rk4_stability(x0,u0)

gp.predict_compare(x0, u_test, model, feedback=False)
#gp.predict_compare(x0, u_test, model, feedback=True, x_ref=x_sp, Q=Q, R=R)

#gp.update_data(X_test, Y_test, int(N*2))
#gp.predict_compare(x0, u_test, model)
#model.predict_compare(x0,u_test)
#model.plot(x0, u_test)

solver_opts = {
               'ipopt.linear_solver' : 'ma27',
               'ipopt.max_cpu_time' : 50,
               'expand' : True,
}

# mpc = MPC(horizon=30*dt, gp=gp, model=model,
#           gp_method='TA',
#           ulb=ulb, uub=uub, xlb=xlb, xub=xub, Q=Q,  R=R, S=S,
#           terminal_constraint=None, costFunc='quad', feedback=True,
#           solver_opts=solver_opts, discrete_method='gp',
#           inequality_constraints=None
#           )
#
#
# x, u = mpc.solve(x0, u0=u0, sim_time=80*dt, x_sp=x_sp, debug=False, noise=True)
# mpc.plot(xnames=['Tank 1 [cm]', 'Tank 2 [cm]','Tank 3 [cm]','Tank 4 [cm]'],
#        unames=['Pump 1 [ml/s]', 'Pump 2 [ml/s]'])

A, B = model.discrete_linearize(x0, u0)
# K, S, E = lqr(A, B, Q, R)
plot_eig(A)
Ad, Bd = gp.discrete_linearize(x0, u0, np.eye(6)*1e-5)
# Kd, Sd, Ed = lqr(Ad, Bd, Q, R)
plot_eig(Ad)
# eig = plot_eig(A + B @ K)
# eig = plot_eig(Ad + Bd @ Kd)
