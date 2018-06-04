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
                * ca.sqrt(2 * g * x[2] + 1e-6) + (gamma1 / A1) * u[0],
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
    


solver_opts = {
                'ipopt.linear_solver' : 'ma27',
                'ipopt.max_cpu_time' : 2,
                'expand' : True,
}

meanFunc = 'zero'
dt = 10.0
Nx = 4
Nu = 2
R = np.eye(Nx) * 1e-5 

# Limits in the training data
ulb = [0., 0.]
uub = [60., 60.]
xlb = [.0, .0, .0, .0]
xub = [30., 30., 30., 30.]

N = 40 # Number of training data

# Create simulation model
model          = Model(Nx=Nx, Nu=Nu, ode=ode, dt=dt, R=R, clip_negative=True)
#X, Y           = model.generate_training_data(N, uub, ulb, xub, xlb, noise=True)
#X_test, Y_test = model.generate_training_data(N, uub, ulb, xub, xlb, noise=True)

# Create GP model
#gp = GP(X, Y, mean_func=meanFunc, normalize=True, xlb=xlb, xub=xub, ulb=ulb, 
#        uub=uub, optimizer_opts=solver_opts, multistart=1)
#gp.update_data(X_test, Y_test, N_new=50)
#gp.save_model('gp_tank')
gp = GP.load_model('gp_tank')
#gp.validate(X_test, Y_test)


# Limits in the MPC problem
ulb = [10., 10.]
uub = [60., 60.]
xlb = [5.0, 5.0, 5.0, 5.0] 
xub = [30., 30., 30., 30.]
x_sp = np.array([14., 14., 14.2, 21.3])
x0 = np.array([9., 10., 8., 18.])
u0 = np.array([45, 55])
u_test = np.full((20, 2), [35, 56]) 

Q = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])
P = np.array([[5, 0, 0, 0],
              [0, 5, 0, 0],
              [0, 0, 5, 0],
              [0, 0, 0, 5]])
R = np.diag([.01, .01])
S = np.diag([.01, .01]) 
model.check_rk4_stability(x0,u0)

#gp.predict_compare(x0, u_test, model, feedback=False, x_ref=x_sp, Q=Q, R=R)
#gp.predict_compare(x0, u_test, model, feedback=True, x_ref=x_sp, Q=Q, R=R)

#gp.update_data(X_test, Y_test, int(N*2))
#gp.predict_compare(x0, u_test, model)
#model.predict_compare(x0,u_test)
#model.plot(x0, u_test)


mpc = MPC(horizon=10*dt, gp=gp, model=model,
          gp_method='TA',
          ulb=ulb, uub=uub, xlb=xlb, xub=xub, Q=Q, P=P, R=R, S=S,
          terminal_constraint=None, costFunc='quad', feedback=False, 
          solver_opts=solver_opts, discrete_method='gp',
          inequality_constraints=None
          )


x, u = mpc.solve(x0, u0=u0, sim_time=20*dt, x_sp=x_sp, debug=False, noise=True)
mpc.plot()
#
#A, B = model.discrete_rk4_linearize(x0, u0)
#K, S, E = lqr(A, B, Q, R)
#Ad, Bd = gp.discrete_linearize(x0, u0, np.eye(6)*1e-5)
#Kd, Sd, Ed = lqr(Ad, Bd, Q, R)
#plot_eig(A)
#eig = plot_eig(A - B @ K)