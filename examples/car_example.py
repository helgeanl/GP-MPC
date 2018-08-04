# -*- coding: utf-8 -*-
"""
@author: Helge-André Langåker
"""
from sys import path
path.append(r"./../") # Add gp_mpc pagkage to path

import numpy as np
import casadi as ca
import scipy.linalg
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
from gp_mpc import Model, GP, MPC, plot_eig, lqr



def ode(x, u, z, p):
    """ Full Bicycle Model
    """
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
    eps = 1e-20                         # Small epsilon to avoid dividing by zero

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
    """ Dynamic equation of Bicycle model for use with GP model
    """
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
            ]
    return  ca.vertcat(*dxdt)


def ode_hybrid(x, u, z, p):
    """ Kinematic equations of Bicycle model for use with hybrid model
    """
    dxdt = [
                u[2],
                u[0]*ca.cos(x[0]) - u[1]*ca.sin(x[0]),
                u[0]*ca.sin(x[0]) + u[1]*ca.cos(x[0])
            ]
    return  ca.vertcat(*dxdt)


def constraint_parameters(x):
    """ Constraint parameters to send to the solver at each iteration
    """
    car_pos = x[4:]
    dist = np.sqrt((car_pos[0] - obs[:,0])**2
                   + (car_pos[1] - obs[:, 1])**2 )
    if min(dist) > 40:
        return np.hstack([car_pos * 1000, [0,0]])

    return obs[np.argmin(dist)]


def inequality_constraints(x, covar, u, eps, par):
    """ Inequality constraints to send to the MPC builder
    """
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

    con_ineq.append(slip_f(x[0], x[1], x[2], u[1]) - slip_max - eps[0])
    con_ineq_ub.append(0)
    con_ineq_lb.append(-np.inf)

    con_ineq.append(slip_min - slip_f(x[0], x[1], x[2], u[1]) - eps[0])
    con_ineq_ub.append(0)
    con_ineq_lb.append(-np.inf)

    con_ineq.append(slip_r(x[0], x[1], x[2]) - slip_max - eps[0])
    con_ineq_ub.append(0)
    con_ineq_lb.append(-np.inf)

    con_ineq.append(slip_min - slip_r(x[0], x[1], x[2]) - eps[0])
    con_ineq_ub.append(0)
    con_ineq_lb.append(-np.inf)

    """ Add road boundry constraints """
    con_ineq.append(x[5] - eps[1])
    con_ineq_ub.append(road_bound)
    con_ineq_lb.append(-road_bound)

    """ Obstacle avoidance """
    Xcg_s = ca.SX.sym('Xcg')
    Ycg_s = ca.SX.sym('Ycg')
    obs_s = ca.SX.sym('obs', 4)
    ellipse = ca.Function('ellipse', [Xcg_s, Ycg_s, obs_s],
                          [ ((Xcg_s - obs_s[0]) / (obs_s[2] + car_length))**2
                           + ((Ycg_s - obs_s[1]) / (obs_s[3] + car_width))**2] )
    con_ineq.append(1 - ellipse(x[4], x[5], par) - eps[2])

    con_ineq_ub.append(0)
    con_ineq_lb.append(-np.inf)

    cons = dict(con_ineq=con_ineq,
                con_ineq_lb=con_ineq_lb,
                con_ineq_ub=con_ineq_ub
            )
    return cons


""" Dynamic Model options"""
dt = 0.05
Nx = 3
Nu = 2
R_n = np.diag([1e-5, 1e-8, 1e-8])

""" Training data options """
N = 200            # Number of training data
N_test = 500       # Number of validation data

normalize = False  # Option to normalize data in GP model

""" Limits in the training data """
ulb = [-.5, -.04]
uub = [.5, .04]
xlb = [10.0, -.6, -.2]
xub = [30.0, .6, .2]

""" Create simulation model """
model_gp = Model(Nx=Nx, Nu=Nu, ode=ode_gp, dt=dt, R=R_n)

""" Generate training and test data  """
X, Y     = model_gp.generate_training_data(N, uub, ulb, xub, xlb, noise=True)
X_test, Y_test = model_gp.generate_training_data(N_test, uub, ulb, xub, xlb, noise=False)

""" Options for hyper-parameter optimization """
solver_opts = {}
solver_opts['ipopt.linear_solver'] = 'ma27' # Faster plugin solver than default
solver_opts['expand']= False                # Choise between SX or MX graph

if 0:
    """ Create GP model estimating dynamics from car model """
    gp = GP(X, Y, ulb=ulb, uub=uub, optimizer_opts=solver_opts, normalize=normalize)
    SMSE, MNLP = gp.validate(X_test, Y_test)
    gp.save_model('models/gp_car')
else:
    """ Load example model """
    gp = GP.load_model('models/gp_car_example')

""" Predict GP open/closed loop """
# Test data
x0 = np.array([13.89, 0.0, 0.0])
x_sp = np.array([13.89, 0., 0.001])
u0 = [0.0, 0.0]
cov0 = np.eye(Nx+Nu)

t = np.linspace(0,20*dt, 20)
u_i = np.sin(0.01*t) * 0
u_test = np.vstack([0.5*u_i, 0.02*u_i]).T

# Penalty matrices for LQR
Q = np.diag([.1, 10., 50.])
R = np.diag([.1, 1])

# Name of states for use with plotting
xnames = [r'$\dot{x}$', r'$\dot{y}$', r'$\dot{\psi}$']

# Predict and plot open loop GP using fixed inputs
gp.predict_compare(x0, u_test, model_gp, feedback=False, x_ref=x_sp, Q=Q, R=R,
                   methods = ['TA','ME'], num_cols=1, xnames=xnames)
# Predict and plot closed loop GP using LQR feedback
gp.predict_compare(x0, u_test, model_gp, feedback=True, x_ref=x_sp, Q=Q, R=R,
                   methods = ['TA', 'ME'], num_cols=1, xnames=xnames)


""" Create hybrid model with state integrator """
Nx = 6
R_n = np.diag([1e-5, 1e-8, 1e-8, 1e-8, 1e-5, 1e-5])
model_hybrid   = Model(Nx=3, Nu=3, ode=ode_hybrid, dt=dt, R=R_n)
model          = Model(Nx=Nx, Nu=Nu, ode=ode, dt=dt, R=R_n)

""" Options for MPC solver"""
solver_opts = {}
#solver_opts['ipopt.linear_solver'] = 'ma27' # Plugin solver from HSL
solver_opts['ipopt.max_cpu_time'] = 20
solver_opts['expand']= True
solver_opts['ipopt.expect_infeasible_problem'] = 'yes'

# Constraint parameters
slip_min = -4.0 * np.pi / 180
slip_max = 4.0 * np.pi / 180
road_bound = 2.0
car_width = 1.2
car_length = 5.

# Position and size of eliptical obsticles [x, y, a, b]
obs = np.array([[20, .3, 0.01, 0.01],
               [60, -0.3, .01, .01],
               [100, 0.3, .01, .01],
               ])

# Limits in the MPC problem
ulb = [-.5, -.04]
uub = [.5, .04]
xlb = [10.0, -.5, -.2, -.3, .0, -10]
xub = [30.0, .5, .2, .3, 500, 10]

# Penalty matrices
Q = np.diag([.001, 5., 1., .1, 1e-10, 1])
R = np.diag([.1, 1])
S = np.diag([1, 10])

# Penalty in soft constraint
lam = 500

# Initial value and set point
x0 = np.array([13.89, 0.0, 0.0, 0.0,.0 , 0.0])
x_sp = np.array([13.89, 0., 0., 0., 100., 0. ])

""" Build MPC object"""
mpc = MPC(horizon=17*dt, model=model,gp=gp, hybrid=model_hybrid,
          discrete_method='hybrid', gp_method='ME',
          ulb=ulb, uub=uub, xlb=xlb, xub=xub, Q=Q, R=R, S=S, lam=lam,
          terminal_constraint=None, costFunc='quad', feedback=True,
          solver_opts=solver_opts,
          inequality_constraints=inequality_constraints, num_con_par=4
          )

""" Simulate measurments and solve MPC problem with constraints for 50 steps"""
x, u = mpc.solve(x0, sim_time=50*dt, x_sp=x_sp, debug=False, noise=True,
                 con_par_func=constraint_parameters)
""" Plot """
mpc.plot()
