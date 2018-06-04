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
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
from gp_mpc import Model, GP, MPC, plot_eig, lqr
import scipy.linalg


def plot_car(x ,y):
    """ Plot the progression of the car in the x-y plane"""

    plt.figure()
    ax = plt.subplot(111)
    ax.axhline(y=road_bound, color='r', linestyle='-')
    ax.axhline(y=0, color='g', linestyle='--')
    ax.axhline(y=-road_bound, color='r', linestyle='-')

    for i in range(np.size(obs,0)):
        ell = Ellipse(xy=obs[i,:2], width=(obs[i,2] + car_length)*2, height=(obs[i,3] + car_width)*2)
        obsticle = Ellipse(xy=obs[i,:2], width=obs[i,2], height=obs[i,3])
        ax.add_artist(ell)
        ax.add_artist(obsticle)
        obsticle.set_facecolor('red')

    ax.plot(x, y, 'b-', linewidth=1.0)
    ax.set_ylabel('y [m]')
    ax.set_xlabel('x [m]')
    plt.show()


def ode(x, u, z, p):
    # Model Parameters (Gao et al., 2014)
    g   = 9.18                          # Gravity [m/s^2]
    m   = 2050                          # Vehicle mass [kg]
    Iz  = 3344                          # Yaw inertia [kg*m^2]
    Cr   = 65000                         # Tyre corning stiffness [N/rad]
    Cf   = 65000                        # Tyre corning stiffness [N/rad]
    mu  = 0.5                           # Tyre friction coefficient
    l   = 4.0                           # Vehicle length
    lf  = 2.0                           # Distance from CG to the front tyre
    lr  = l - lf                        # Distance from CG to the rear tyre
    Fzf = lr * m * g / (2 * l)          # Vertical load on front wheels
    Fzr = lf * m * g / (2 * l)          # Vertical load on rear wheels
    eps = 1e-10                        # Small epsilon to avoid dividing by zero
#    x[0] = ca.fmax(x[0], 1e-10)
    dxdt = [
                1/m * (m*x[1]*x[2] + 2*mu*Fzf*u[0] + 2*Cf*u[1]**2
                    - 2*Cf*u[1] * (x[1] + lf*x[2]) / (x[0] + eps) + 2*mu*Fzr*u[0]),
                1/m * (-m*x[0]*x[2] + 2*mu*Fzf*u[1]*u[0]
                    + 2*Cf*(x[1] + lf*x[2]) / (x[0] + eps) - 2*Cf*u[1]
                    + 2*Cr*(x[1] - lf*x[2]) / (x[0] + eps)),
                1/Iz * (2*lf*mu*Fzf*u[0]*u[1] + 2*lf*Cf*(x[1] + lf*x[2]) / (x[0] + eps)
                    - 2*lf*Cf*u[1] - 2*lr*Cr*(x[1] - lf*x[2]) / (x[0] + eps)),
                x[2],
                x[0]*ca.cos(x[3]) - x[1]*ca.sin(x[3]),
                x[0]*ca.sin(x[3]) + x[1]*ca.cos(x[3])
            ]
    return  ca.vertcat(*dxdt)


def ode_gp(x, u, z, p):
    # Model Parameters (Gao et al., 2014)
    g   = 9.18                          # Gravity [m/s^2]
    m   = 2050                          # Vehicle mass [kg]
    Iz  = 3344                          # Yaw inertia [kg*m^2]
    Cr   = 65000                         # Tyre corning stiffness [N/rad]
    Cf   = 65000                        # Tyre corning stiffness [N/rad]
    mu  = 0.5                           # Tyre friction coefficient
    l   = 4.0                           # Vehicle length
    lf  = 2.0                           # Distance from CG to the front tyre
    lr  = l - lf                        # Distance from CG to the rear tyre
    Fzf = lr * m * g / (2 * l)          # Vertical load on front wheels
    Fzr = lf * m * g / (2 * l)          # Vertical load on rear wheels
    eps = 1e-10                          # Small epsilon to avoid dividing by zero

    dxdt = [
                1/m * (m*x[1]*x[2] + 2*mu*Fzf*u[0] + 2*Cf*u[1]**2
                    - 2*Cf*u[1] * (x[1] + lf*x[2]) / (x[0] + eps) + 2*mu*Fzr*u[0]),
                1/m * (-m*x[0]*x[2] + 2*mu*Fzf*u[1]*u[0]
                    + 2*Cf*(x[1] + lf*x[2]) / (x[0] + eps) - 2*Cf*u[1]
                    + 2*Cr*(x[1] - lf*x[2]) / (x[0] + eps)),
                1/Iz * (2*lf*mu*Fzf*u[0]*u[1] + 2*lf*Cf*(x[1] + lf*x[2]) / (x[0] + eps)
                    - 2*lf*Cf*u[1] - 2*lr*Cr*(x[1] - lf*x[2]) / (x[0] + eps)),
#                x[2],
#                x[0]*ca.cos(x[3]) - x[1]*ca.sin(x[3]),
#                x[0]*ca.sin(x[3]) + x[1]*ca.cos(x[3])
            ]
    return  ca.vertcat(*dxdt)


def ode_hybrid(x, u, z, p):
    # Model Parameters (Gao et al., 2014)
    g   = 9.18                          # Gravity [m/s^2]
    m   = 2050                          # Vehicle mass [kg]
    Iz  = 3344                          # Yaw inertia [kg*m^2]
    Cr   = 65000                         # Tyre corning stiffness [N/rad]
    Cf   = 65000                        # Tyre corning stiffness [N/rad]
    mu  = 0.5                           # Tyre friction coefficient
    l   = 4.0                           # Vehicle length
    lf  = 2.0                           # Distance from CG to the front tyre
    lr  = l - lf                        # Distance from CG to the rear tyre
    Fzf = lr * m * g / (2 * l)          # Vertical load on front wheels
    Fzr = lf * m * g / (2 * l)          # Vertical load on rear wheels
    eps = 1e-10                         # Small epsilon to avoid dividing by zero

    dxdt = [
                0,
                0,
                0,
                x[2],
                x[0]*ca.cos(x[3]) - x[1]*ca.sin(x[3]),
                x[0]*ca.sin(x[3]) + x[1]*ca.cos(x[3])
            ]
    return  ca.vertcat(*dxdt)


def constraint_parameters(x):
    car_pos = x[4:]
    dist = np.sqrt((car_pos[0] - obs[:,0])**2
                   + (car_pos[1] - obs[:, 1])**2 )
    if min(dist) > 40:
        return np.hstack([car_pos * 1000, [0,0]])

    return obs[np.argmin(dist)]


def inequality_constraints(x, covar, u, eps, par):
    con_ineq = []
    con_ineq_lb = []
    con_ineq_ub = []

    """ Slip angle constraint """
    dx_s = ca.SX.sym('dx')
    dy_s = ca.SX.sym('dy')
    dpsi_s = ca.SX.sym('dpsi')
    delta_f_s = ca.SX.sym('delta_f')
    lf  = 2.0
    lr  = 2.0

    slip_f = ca.Function('slip_f', [dx_s, dy_s, dpsi_s, delta_f_s],
                         [(dy_s + lf*dpsi_s)/(dx_s + 1e-6)   - delta_f_s])
    slip_r = ca.Function('slip_r', [dx_s, dy_s, dpsi_s],
                         [(dy_s - lr*dpsi_s)/(dx_s + 1e-6)])

    con_ineq.append(slip_f(x[0], x[1], x[2], u[1]) - slip_max - eps)
    con_ineq_ub.append(0)
    con_ineq_lb.append(-np.inf)

    con_ineq.append(slip_min - slip_f(x[0], x[1], x[2], u[1]) - eps)
    con_ineq_ub.append(0)
    con_ineq_lb.append(-np.inf)

    con_ineq.append(slip_r(x[0], x[1], x[2]) - slip_max - eps)
    con_ineq_ub.append(0)
    con_ineq_lb.append(-np.inf)

    con_ineq.append(slip_min - slip_r(x[0], x[1], x[2]) - eps)
    con_ineq_ub.append(0)
    con_ineq_lb.append(-np.inf)

    """ Add road boundry constraints """
    con_ineq.append(x[5])
    con_ineq_ub.append(road_bound)
    con_ineq_lb.append(-road_bound)

    """ Obstacle avoidance """
    Xcg_s = ca.SX.sym('Xcg')
    Ycg_s = ca.SX.sym('Ycg')
    obs_s = ca.SX.sym('obs', 4)
    ellipse = ca.Function('ellipse', [Xcg_s, Ycg_s, obs_s],
                          [ ((Xcg_s - obs_s[0]) / (obs_s[2] + car_length))**2
                           + ((Ycg_s - obs_s[1]) / (obs_s[3] + car_width))**2] )
    con_ineq.append(1 - ellipse(x[4], x[5], par) - eps)

    con_ineq_ub.append(0)
    con_ineq_lb.append(-np.inf)

    cons = dict(con_ineq=con_ineq,
                con_ineq_lb=con_ineq_lb,
                con_ineq_ub=con_ineq_ub
            )
    return cons



solver_opts = {}
solver_opts['ipopt.linear_solver'] = 'ma27'
solver_opts['ipopt.max_cpu_time'] = 2
#solver_opts['ipopt.max_iter'] = 75
solver_opts['expand']= True
solver_opts['ipopt.expect_infeasible_problem'] = 'yes'
#solver_opts['print_time'] = True
#solver_opts['ipopt.print_level'] = 5


dt = 0.05
Nx = 3
Nu = 2
R_n = np.diag([1e-5, 1e-8, 1e-8])


# Limits in the training data
ulb = [-.5, -.05]
uub = [.5, .05]
#xlb = [10.0, -.8, -.5, -.5, .0,  -1.]
xlb = [10.0, -.6, -.2]
#xub = [30.0, .8, .5, .5, 10, 1]
xub = [30.0, .6, .2]

N = 50 # Number of training data

# Create simulation model
model_gp       = Model(Nx=Nx, Nu=Nu, ode=ode_gp, dt=dt, R=R_n)
X, Y           = model_gp.generate_training_data(N, uub, ulb, xub, xlb, noise=True)
X_test, Y_test = model_gp.generate_training_data(100, uub, ulb, xub, xlb, noise=False)

#gp = GP(X, Y, ulb=ulb, uub=uub, optimizer_opts=solver_opts, normalize=True)
#gp.validate(X_test, Y_test)
#gp.update_data(X_test, Y_test, N_new=100)
#X_test, Y_test = model_gp.generate_training_data(100, uub, ulb, xub, xlb, noise=True)
#gp.save_model('gp_car_150_dt_005_reduced_normalized_latin')
gp = GP.load_model('gp_car_150_reduced_normalized_latin')
gp.validate(X_test, Y_test)


# Create hybrid model
Nx = 6
R_n = np.diag([1e-5, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7])
model_hybrid   = Model(Nx=Nx, Nu=Nu, ode=ode_hybrid, dt=dt, R=R_n)
model          = Model(Nx=Nx, Nu=Nu, ode=ode, dt=dt, R=R_n)


# Test data
x0 = np.array([13.89, 0.0, 0.0, 0.0,.0 , 0.0])
#x0 = np.array([13.89, 0.0, 0.0])
u0 = [0.0, 0.0]
cov0 = np.eye(Nx+Nu)
u_test = np.zeros((20, 2))

## Create GP model
#gp = GP(X, Y, xlb=xlb, xub=xub, ulb=ulb, uub=uub, optimizer_opts=solver_opts, normalize=False)
#gp.save_model('gp_car')
#gp = GP.load_model('gp_car')
#gp.validate(X_test, Y_test)
#gp.predict_compare(x0, u_test, model)
A, B = model.discrete_linearize(x0, u0)
model.check_rk4_stability(x0,u0)
#eig = plot_eig(A)
#print(eig)

# Limits in the MPC problem
ulb = [-.5, -.034]
uub = [.5, .034]
xlb = [10.0, -.5, -.15, -.3, .0, -10]
xub = [30.0, .5, .15, .3, 500, 10]
x_sp = np.array([13.89, 0., 0., 0., 100., 0. ])
#x_sp = np.array([13.89, 0., 0.])

# Constraint parameters
slip_min = -4.0 * np.pi / 180
slip_max = 4.0 * np.pi / 180
road_bound = 2.0
car_width = 1.2  #1.0
car_length = 5. #10.0
obs = np.array([[20, .3, 0.01, 0.01],
               [60, -0.3, .01, .01],
#               [100, 0.3, .01, .01],
               ])

#obs = np.array([[20,.0,0.01,0.01]
#               ])


# Penalty values
P = np.diag([.0, 50., 10, .1, 0, 10])
#Q = np.diag([.0, 5., 1., .01, 0, .1])
Q = np.diag([.01, 5., 1., .1, 0.0, 1])
#Q = np.diag([.1, 10., 10.])
R = np.diag([1, 1])
S = np.diag([1, 10])
lam = 100

Bd = np.vstack([np.eye(3), np.zeros((3,3))])
#gp.predict_compare(x0, u_test, model, feedback=False, x_ref=x_sp, Q=Q, R=R)
#gp.predict_compare(x0, u_test, model, feedback=True, x_ref=x_sp, Q=Q, R=R)

mpc = MPC(horizon=2*dt, model=model,gp=gp, hybrid=model_hybrid, Bd = Bd,
          discrete_method='hybrid', gp_method='TA',
          ulb=ulb, uub=uub, xlb=xlb, xub=xub, Q=Q, P=P, R=R, S=S, lam=lam,
          terminal_constraint=None, costFunc='quad', feedback=False,
          solver_opts=solver_opts,
          inequality_constraints=inequality_constraints, num_con_par=4
          )


x, u = mpc.solve(x0, sim_time=200*dt, x_sp=x_sp, debug=False, noise=False,
                 con_par_func=constraint_parameters)
mpc.plot()
plot_car(x[:, 4], x[:, 5])
#u1 = u[:20,:]
#model.predict_compare(x[0], u1)

## Use previous data to train GP
#X = x[:-1,:]
#Y = x[1:,:]
#Z = np.hstack([X, u])
#Z1 = Z[::3,:3]
#Y1 = Y[::3,:3]
#Z2 = Z[2::3,:3]
#Y2 = Y[2::3,:3]


## Create GP model
#solver_opts['expand']= False
#gp = GP(Z[:,:3], Y[:,:3], ulb=ulb, uub=uub, optimizer_opts=solver_opts, normalize=True)
#gp.save_model('gp_car_300_reduzed_normalized')
##gp = GP.load_model('gp_car')
#gp.validate(Z2, Y2)
#gp.predict_compare(x0, u_test, model)

#mpc_gp = MPC(horizon=2*dt, gp=gp, model=model,
#          discrete_method='gp', gp_method='TA',
#          ulb=ulb, uub=uub, xlb=xlb, xub=xub, Q=Q, P=P, R=R, S=S, lam=lam,
#          terminal_constraint=None, costFunc='quad', feedback=False,
#          solver_opts=solver_opts,
#          inequality_constraints=inequality_constraints, num_con_par=4
#          )
#
#
#x, u = mpc_gp.solve(x0, sim_time=3*dt, x_sp=x_sp, debug=False, noise=False,
#                 con_par_func=constraint_parameters)
