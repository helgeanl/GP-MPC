# Generate simulation data for regression model
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sys import path
path.append(r"C:\Users\helgeanl\Google Drive\NTNU\Masteroppgave\casadi-py27-v3.3.0")

import pyDOE
import pylab
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

# Specifications
# Discrete Time Model
ndstate = 4                             # Number of states
nastate = 0                             # Number of algebraic equations
ninput  = 2                             # Number of inputs
deltat  = 30                             # Time of one control interval 3s

training = True                         # True: generate training data
optimize = False                        # Optimize hyperperameters
sim = False                             # True: simualate and plot system
simTime = 300                           # Simulation time in seconds


# Regression data
npoints = 30                           # Number of data points generated
u_min = np.array([0., 0.])              # lower bound of control inputs
u_max = np.array([100., 100.])          # upper bound of control inputs
x_min = np.array([0., 0., 0., 0.])      # lower bound of expected minimum state
x_max = np.array([80., 80., 80., 80])   # upper bound of expected minimum state
R = np.diag([1e-3, 1e-3, 1e-3, 1e-3])   # noise covariance matrix


def integrate_system(ndstate, nastate, u, t0, tf, x0):

    # Model Parameters
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

    # Differential states
    xd = ca.SX.sym("xd", ndstate)  # vector of states [h1,h2,h3,h4]

    # Initial conditions
    xDi = x0

    ode = ca.vertcat(
        -a1 / A1 * ca.sqrt(2 * g * xd[0]) + a3 / A1 * ca.sqrt(2 * g * xd[2]) + gamma1 / A1 * u[0],
        -a2 / A2 * ca.sqrt(2 * g * xd[1]) + a4 / A2 * ca.sqrt(2 * g * xd[3]) + gamma2 / A2 * u[1],
        -a3 / A3 * ca.sqrt(2 * g * xd[2]) + (1 - gamma2) / A3 * u[1],
        -a4 / A4 * ca.sqrt(2 * g * xd[3]) + (1 - gamma1) / A4 * u[0])

    dae = {'x': xd, 'ode': ode}

    # Create a DAE system solver
    opts = {}
    opts['abstol'] = 1e-10  # abs. tolerance
    opts['reltol'] = 1e-10  # rel. tolerance
    opts['t0'] = t0
    opts['tf'] = tf
    Sim = ca.integrator('Sim', 'idas', dae, opts)
    res = Sim(x0=xDi)
    x_current = pylab.array(res['xf'])

    return x_current


# -----------------------------------------------------------------------------
# Simulate and plot system
# -----------------------------------------------------------------------------
def sim_system(x0, u, simTime, deltat, noise=False):
    simPoints = int(simTime / deltat)
    # Predefine matrix to collect control inputs
    u_matrix = u

    # Initial state of the system
    x = x0

    # Predefine matrix to collect noisy state outputs
    Y_sim = np.zeros((simPoints, ndstate))

    for dt in range(simPoints):
        t0i = 0.                 # start time of integrator
        tfi = deltat             # end time of integrator
        u_s = u_matrix[dt, :]    # control input for simulation

        # simulate system
        x = integrate_system(ndstate, nastate, u_s, t0i, tfi, x)[:, 0]

        # save simulated state with normal white noise added
        if noise:
            Y_sim[dt, :] = x + np.random.multivariate_normal(np.zeros((ndstate)), R)
        else:
            Y_sim[dt, :] = x

    return Y_sim


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
        t0i = 0.                 # start time of integrator
        tfi = deltat             # end time of integrator
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
        np.savetxt('../data/' + 'X_matrix_tank', X_mat)    # Save input matrix  as text file
        np.savetxt('../data/' + 'Y_matrix_tank', Y_mat)    # Save output matrix as text file

    if sim is True:
        # Plot simulation
        simPoints = simTime / deltat
        u_matrix = np.zeros((simPoints, 2))
        u_matrix[:, 0] = 50
        u_matrix[:, 1] = 50
        x0 = np.array([10, 20, 30, 40])

        t = np.linspace(0.0, 300.0, 100)
        Y_sim = sim_system(x0, u_matrix, simTime, deltat)
        plt.figure()
        plt.clf()
        for i in range(4):
            plt.subplot(2, 2, i + 1)
            plt.title('A tale of 2 subplots')
            plt.plot(t, Y_sim[:, i], 'b-')
            plt.ylabel('X')
        plt.show()


if __name__ == "__main__":
    main()
