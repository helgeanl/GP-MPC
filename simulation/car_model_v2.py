# Generate simulation data for regression model
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sys import path
path.append(r"C:\Users\helgeanl\Google Drive\NTNU\Masteroppgave\casadi-py36-v3.4.0")

import pyDOE
import pylab
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

# Specifications
# Discrete Time Model
ndstate = 6                             # Number of states
nastate = 0                             # Number of algebraic equations
ninput  = 3                             # Number of inputs
dt      = 0.03                          # Time of one control interval 

training = True                        # True: generate training data
sim = False                              # True: simualate and plot system
simTime = 10*dt # 300                           # Simulation time in seconds


# Regression data
npoints = 10                            # Number of data points generated
u_min = np.array([-.5, -.5, -.1])       # Lower bound of control inputs
u_max = np.array([.5, .5, .1])          # Upper bound of control inputs
x_min = np.array([ 10,-.5, -2.,
                  -2., 0., 0.])       # Lower bound of expected minimum state 
x_max = np.array([20, .5, 2.,
                  2., .5, 10.])        # Upper bound of expected minimum state
R     = np.diag([1e-3, 1e-3, 1e-3, 
                 1e-3, 1e-3, 1e-3])     # Noise covariance matrix


def integrate_system(ndstate, nastate, u, t0, tf, x0):

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


    # Differential states
    x = ca.SX.sym("x", ndstate)  # vector of states 
    
    ode = ca.vertcat(
        1/m * (m*x[1]*x[2] + 2*mu*Fzf*u[0] + 2*Cf*u[2]**2 - 
            2*Cf*u[2] * (x[1] + lf*x[2]) / x[0] + 2*mu*Fzr*u[1]),
        1/m * (-m*x[0]*x[2] + 2*mu*Fzf*u[2]*u[0] + 
            2*Cf*(x[1] + lf*x[2]) / x[0] - 2*Cf*u[2] +
            2*Cr*(x[1] - lf*x[2]) / x[0]),
        1/Iz * (2*lf*mu*Fzf*u[0]*u[2] + 2*lf*Cf*(x[1] + lf*x[2]) / x[0] - 
            2*lf*Cf*u[2] - 2*lr*Cr*(x[1] - lf*x[2]) / x[0]),
        x[2],
        x[0]*ca.cos(x[3]) - x[1]*ca.sin(x[3]),
        x[0]*ca.sin(x[3]) + x[1]*ca.cos(x[3])
    )

    dae = {'x': x, 'ode': ode}

    # Create a DAE system solver
    opts = {}
    opts['abstol'] = 1e-10  # abs. tolerance
    opts['reltol'] = 1e-10  # rel. tolerance
    opts['t0'] = t0
    opts['tf'] = tf
    
    Sim = ca.integrator('Sim', 'idas', dae, opts)
    res = Sim(x0=x0)
    x_current = pylab.array(res['xf'])

    return x_current


# -----------------------------------------------------------------------------
# Simulate system
# -----------------------------------------------------------------------------
def sim_system(x0, u, simTime, dt, noise=False):
    simPoints = int(simTime / dt)
    # Predefine matrix to collect control inputs
    u_matrix = u
    
    # Initial state of the system
    x = x0
    
    # Predefine matrix to collect noisy state outputs
    Y_sim = np.zeros((simPoints, ndstate))

    for t in range(simPoints):
        t0i = 0.                 # start time of integrator
        tfi = dt                 # end time of integrator
        u_s = u_matrix[t, :]    # control input for simulation
        print('Sim u:')
        print(u_s)
        # simulate system
        x = integrate_system(ndstate, nastate, u_s, t0i, tfi, x)[:, 0]
        print('Sim y:')
        print(x)
        # save simulated state with normal white noise added
        if noise:
            Y_sim[t, :] = x + np.random.multivariate_normal(np.zeros((ndstate)), R)
        else:
            Y_sim[t, :] = x

    return Y_sim


# -----------------------------------------------------------------------------
# Simulate system
# -----------------------------------------------------------------------------
def onestep_system(x0, u, dt, noise=False):

    # Initial state of the system
    x = x0
    
    t0i = 0.                 # start time of integrator
    tfi = dt                 # end time of integrator

    # simulate system
    x = integrate_system(ndstate, nastate, u, t0i, tfi, x)[:, 0]

    # save simulated state with normal white noise added
    if noise:
        y = x + np.random.multivariate_normal(np.zeros((ndstate)), R)
    else:
        y = x

    return y

def generate_training_data():
    # Predefine matrix to collect control inputs
    u_mat = np.zeros((npoints, ninput))
    # Predefine matrix to collect state inputs
    X_mat = np.zeros((npoints, ndstate))
    # Predefine matrix to collect noisy state outputs
    Y_mat = np.zeros((npoints, ndstate))

    # Create control input design using a latin hypecube
    # Latin hypercube design for unit cube [0,1]^ndstate
    u_matrix = pyDOE.lhs(ninput, samples=npoints, criterion='maximin')

    # Scale control inputs to correct range
    for k in range(npoints):
        u_mat[k, :] = u_matrix[k, :] * (u_max - u_min) + u_min

    # Create state input design using a latin hypecube
    # Latin hypercube design for unit cube [0,1]^ndstate
    X_mat = pyDOE.lhs(ndstate, samples=npoints, criterion='maximin')

    # Scale state inputs to correct range
    for k in range(npoints):
        X_mat[k, :] = X_mat[k, :] * (x_max - x_min) + x_min

    for un in range(npoints):
        t0i = 0.              # start time of integrator
        tfi = dt              # end time of integrator
        u_s = u_mat[un, :]    # control input for simulation
        x_s = X_mat[un, :]    # state input for simulation

        # simulate system with x_s and u_s inputs for deltat time
        x_output = integrate_system(ndstate, nastate, u_s, t0i, tfi, x_s)[:, 0]
        # save simulated state with normal white noise added
        Y_mat[un, :] = x_output + np.random.multivariate_normal(np.zeros((ndstate)), R)

    # Concatenate inputs to obtain overall input to GP model
    X_mat = np.hstack([X_mat, u_mat])
    return X_mat, Y_mat


def main():
    # -----------------------------------------------------------------------------
    # Generation of training data
    # -----------------------------------------------------------------------------
    if training is True:
        X_mat, Y_mat = generate_training_data()
        np.savetxt('../data/' + 'X_matrix_car', X_mat)    # Save input matrix  as text file
        np.savetxt('../data/' + 'Y_matrix_car', Y_mat)    # Save output matrix as text file
        
        X_mat, Y_mat = generate_training_data()
        np.savetxt('../data/' + 'X_matrix_test_car', X_mat)    # Save input matrix  as text file
        np.savetxt('../data/' + 'Y_matrix_test_car', Y_mat)    # Save output matrix as text file

    if sim is True:
        # Plot simulation
        simPoints = int(simTime / dt)
        u_matrix = np.zeros((simPoints, 3))
        u_matrix[:, 0] = 0.5
        u_matrix[:, 1] = 0.5
        u_matrix[:, 2] = 0.001
        x0 = np.array([20, 0, 0, 0, 0 , 0])
        
        t = np.linspace(0.0, simTime, simPoints)
        Y_sim = sim_system(x0, u_matrix, simTime, dt)
        
        plt.figure()
        plt.clf()
        for i in range(6):
            plt.subplot(3, 2, i + 1)
            plt.plot(t, Y_sim[:, i], 'b-')
            plt.ylabel('X' + str(i + 1))
        plt.show()


if __name__ == "__main__":
    main()
