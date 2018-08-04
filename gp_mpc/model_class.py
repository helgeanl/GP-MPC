# -*- coding: utf-8 -*-
"""
Dynamic System Model
Copyright (c) 2018, Helge-André Langåker
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import pyDOE
import numpy as np
import casadi as ca
import scipy.linalg
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


class Model:
    def __init__(self, Nx, Nu, ode, dt, R=None,
                 alg=None, alg_0=None, Nz=0, Np=0,
                 opt=None, clip_negative=False):
        """ Initialize dynamic model

        # Arguments:
            Nx:   Number of states
            Nu:   Number of inputs
            ode:  ode(x, u, z, p)
            dt:   Sampling time

        # Arguments (optional):
            R:   Noise covariance matrix (Ny, Ny)
            alg: alg(x, z, u)
            alg_0: Initial value of algebraic variables
            Nz:  Number of algebraic states
            Np:  Number of parameters
            opt: Options dict to pass to the IDEAS integrator
            clip_negative: If true, clip negative simulated outputs to zero
        """

        # Create a default noise covariance matrix
        if R is None:
            self.__R = np.eye(self.__Ny) * 1e-3
        else:
            self.__R = R

        self.__dt   = dt
        self.__Nu = Nu
        self.__Nx = Nx
        self.__Nz = Nz
        self.__Np = Np
        self.__clip_negative = clip_negative

        """ Create integrator """
        # Integrator options
        options = {
            "abstol" : 1e-5,
            "reltol" : 1e-9,
            "max_num_steps": 100,
            "tf" : dt,
        }
        if opt is not None:
            options.update(opt)

        x = ca.MX.sym('x', Nx)
        u = ca.MX.sym('u', Nu)
        z = ca.MX.sym('z', Nz)
        p = ca.MX.sym('p', Np)
        par = ca.vertcat(u, p)

        dae = {'x': x, 'ode': ode(x,u,z,p), 'p':par}
        if alg is not None:
            self.__alg0 = ca.Function('alg_0', [x, u],
                                      [alg_0(x, u)])
            dae.update({'z':z, 'alg': alg(x, z, u)})
            self.Integrator = ca.integrator('DEA_Integrator', 'idas', dae, options)
        else:
            self.Integrator = ca.integrator('ODE_Integrator', 'cvodes', dae, options)

        #TODO: Fix discrete DAE model
        if alg is None:
            """ Create discrete RK4 model """
            ode_casadi = ca.Function("ode", [x, u, p], [ode(x,u,z,p)])
            k1 = ode_casadi(x, u, p)
            k2 = ode_casadi(x + dt/2*k1, u, p)
            k3 = ode_casadi(x + dt/2*k2, u, p)
            k4 = ode_casadi(x + dt*k3,u, p)
            xrk4 = x + dt/6*(k1 + 2*k2 + 2*k3 + k4)
            self.rk4 = ca.Function("ode_rk4", [x, u, p], [xrk4])

            # Jacobian of continuous system
            self.__jac_x = ca.Function('jac_x', [x, u, p],
                                       [ca.jacobian(ode_casadi(x,u,p), x)])
            self.__jac_u = ca.Function('jac_x', [x, u, p],
                                       [ca.jacobian(ode_casadi(x,u,p), u)])

            # Jacobian of discrete RK4 system
            self.__discrete_rk4_jac_x = ca.Function('jac_x', [x, u, p],
                                        [ca.jacobian(self.rk4(x,u,p), x)])
            self.__discrete_rk4_jac_u = ca.Function('jac_x', [x, u, p],
                                        [ca.jacobian(self.rk4(x,u,p), u)])

        # Jacobian of exact discretization
        self.__discrete_jac_x = ca.Function('jac_x', [x, u, p],
                                   [ca.jacobian(self.Integrator(x0=x,
                                             p=ca.vertcat(u,p))['xf'], x)])

        self.__discrete_jac_u = ca.Function('jac_u', [x, u, p],
                                    [ca.jacobian(self.Integrator(x0=x,
                                              p=ca.vertcat(u,p))['xf'], u)])


    def linearize(self, x0, u0, p0=[]):
        """ Linearize the continuous system around the operating point
            dx/dt = Ax + Bu
        # Arguments:
            x0: State vector
            u0: Input vector
            p0: Parameter vector (optional)
        """
        A = np.array(self.__jac_x(x0, u0, p0))
        B = np.array(self.__jac_u(x0, u0, p0))
        return A, B


    def discrete_linearize(self, x0, u0, p0=[]):
        """ Linearize the exact discrete system around the operating point
            x[k+1] = Ax[k] + Bu[k]
        # Arguments:
            x0: State vector
            u0: Input vector
            p0: Parameter vector (optional)
        """
        Ad = np.array(self.__discrete_jac_x(x0, u0, p0))
        Bd = np.array(self.__discrete_jac_u(x0, u0, p0))
        return Ad, Bd


    def discrete_rk4_linearize(self, x0, u0, p0=[]):
        """ Linearize the discrete rk4 system around the operating point
            x[k+1] = Ax[k] + Bu[k]
        # Arguments:
            x0: State vector
            u0: Input vector
            p0: Parameter vector (optional)
        """
        Ad = np.array(self.__discrete_rk4_jac_x(x0, u0, p0))
        Bd = np.array(self.__discrete_rk4_jac_u(x0, u0, p0))
        return Ad, Bd


    def rk4_jacobian_x(self, x0, u0, p0=[]):
        """ Return state jacobian evaluated at the operating point
            x[k+1] = Ax[k] + Bu[k]
        # Arguments:
            x0: State vector
            u0: Input vector
            p0: Parameter vector (optional)
        """
        return self.__discrete_rk4_jac_x(x0, u0, p0)


    def rk4_jacobian_u(self, x0, u0, p0=[]):
        """ Return input jacobian evaluated at the operating point
            x[k+1] = Ax[k] + Bu[k]
        # Arguments:
            x0: State vector
            u0: Input vector
            p0: Parameter vector (optional)
        """
        return self.__discrete_rk4_jac_u(x0, u0, p0)


    def check_rk4_stability(self, x0, u0, d=.1, plot=False):
        """ Check if Runga Kutta 4 method is stable around operating point

        # Return True if stable, False if not stable
        """
        A, B = self.linearize(x0, u0, p0=[])
        eigenvalues, eigenvec = scipy.linalg.eig(A)
        h = self.sampling_time()
        for eig in eigenvalues:
            R = 1 + h*eig + (h*eig)**2/2 + (h*eig)**3/6 + (h*eig)**4/24
            if np.abs(R) >= 1:
                return False
#        if plot:
#            h = d
##            N = 1000;
##            th = np.linspace(0, 2*np.pi, N);
##            r = np.exp(1j*th);
##            f = lambda r: 1 + h*r + (h*r)**2/2 + (h*r)**3/6 + (h*r)**4/24
#            plt.figure()
#            x = np.arange(-3.0, 3.0, 0.01)
#            y = np.arange(-3.0, 3.0, 0.01)
#            X, Y = np.meshgrid(x, y)
#            print(h)
#            z = X + 1j*Y;
#            R = 1 + h*z + (h*z)**2/2 + (h*z)**3/6 + (h*z)**4/24
#            print(R.shape)
#            zlevel4 = abs(R);
#            plt.contour(x,y, zlevel4)
#            plt.show()
        return True


    def sampling_time(self):
        """ Get the sampling time
        """
        return self.__dt


    def size(self):
        """ Get the size of the model

        # Returns:
                Nx: Number of states
                Nu: Number of inputs
                Np: Number of parameters
        """
        return self.__Nx, self.__Nu, self.__Np


    def integrate(self, x0, u, p):
        """ Integrate one time sample dt

        # Arguments:
            x0: Initial state vector
            u: Input vector
            p: Parameter vector
        # Returns:
            x: Numpy array with x at t0 + dt
        """
        par=ca.vertcat(u, p)
        if self.__Nz is not 0:
            z0 = self.__alg0(x0, u)
            out = self.Integrator(x0=x0, p=u, z0=z0)
        else:
            out = self.Integrator(x0=x0, p=par)
        return np.array(out["xf"]).flatten()

#TODO: Fix this or remove
    def set_method(self, method='exact'):
        """ Select wich discrete time method to use """


    def sim(self, x0, u, p=None, noise=False):
        """ Simulate system

        # Arguments:
            x0: Initial state (Nx, 1)
            u: Input matrix with the input for each timestep in the
                simulation horizon (Nt, Nu)
            p: Parameter matrix with the parameters for each timestep
                in the simulation horizon (Nt, Np)
            noise: If True, add gaussian noise using the noise covariance matrix

        # Output:
            Y_sim: Matrix with the simulated outputs (Nt, Ny)
        """

        Nt = np.size(u, 0)

        # Initial state of the system
        x = x0

        # Predefine matrix to collect noisy state outputs
        Y = np.zeros((Nt, self.__Nx))

        for t in range(Nt):
            u_t = u[t, :]    # control input for simulation
            if p is not None:
                p_t = p[t, :]    # parameter at step t
            else:
                p_t = []
            try:
                x = self.integrate(x, u_t, p_t).flatten()
            except RuntimeError:
                print('----------------------------------------')
                print('** System unstable, simulator crashed **')
                print('** t: %d **' % t)
                print('----------------------------------------')
                return Y
            Y[t, :] = x

            # Add normal white noise to state outputs
            if noise:
                Y[t, :] += np.random.multivariate_normal(
                                    np.zeros((self.__Nx)), self.__R)

            # Limit values to above 1e-8 to avvoid to avvoid numerical errors
            if self.__clip_negative:
                if np.any(Y < 0):
                    print('Clipping negative values in simulation!')
                    Y = Y.clip(min=1e-6)
        return Y


    def generate_training_data(self, N, uub, ulb, xub, xlb,
                               pub=None, plb=None, noise=True):
        """ Generate training data using latin hypercube design

        # Arguments:
            N:   Number of data points to be generated
            uub: Upper input range (Nu,1)
            ulb: Lower input range (Nu,1)
            xub: Upper state range (Ny,1)
            xlb: Lower state range (Ny,1)

        # Returns:
            Z: Matrix (N, Nx + Nu) with state x and inputs u at each row
            Y: Matrix (N, Nx) where each row is the state x at time t+dt,
                with the input from the same row in Z at time t.
        """
        # Make sure boundry vectors are numpy arrays
        uub = np.array(uub)
        ulb = np.array(ulb)
        xub = np.array(xub)
        xlb = np.array(xlb)

        # Predefine matrix to collect noisy state outputs
        Y = np.zeros((N, self.__Nx))

        # Create control input design using a latin hypecube
        # Latin hypercube design for unit cube [0,1]^Nu
        if self.__Nu > 0:
            U = pyDOE.lhs(self.__Nu, samples=N, criterion='maximin')
             # Scale control inputs to correct range
            for k in range(N):
                U[k, :] = U[k, :] * (uub - ulb) + ulb
        else:
            U = []

        # Create state input design using a latin hypecube
        # Latin hypercube design for unit cube [0,1]^Ny
        X = pyDOE.lhs(self.__Nx, samples=N, criterion='maximin')

        # Scale state inputs to correct range
        for k in range(N):
            X[k, :] = X[k, :] * (xub - xlb) + xlb

        # Create parameter matrix
        par = pyDOE.lhs(self.__Np, samples=N)
        if pub is not None:
            for k in range(N):
                par[k, :] = par[k, :] * (pub - plb) + plb

        for i in range(N):
            if self.__Nu > 0:
                u_t = U[i, :]    # control input for simulation
            else:
                u_t = []
            x_t = X[i, :]    # state input for simulation
            p_t = par[i, :]    # parameter input for simulation

            # Simulate system with x_t and u_t inputs for deltat time
            Y[i, :] = self.integrate(x_t, u_t, p_t)

            # Add normal white noise to state outputs
            if noise:
                Y[i, :] += np.random.multivariate_normal(
                                np.zeros((self.__Nx)), self.__R)

        # Concatenate previous states and inputs to obtain overall input to GP model
        if self.__Nu > 0:
            Z = np.hstack([X, U])
        else:
            Z = X
        return Z, Y


    def plot(self, x0, u, numcols=2):
        """ Simulate and plot model

        # Arguments:
            x0: Initial state
            u: Matrix with inputs for all time steps (Nt, Nu)
            numcols: Number of columns in the plot
        """
        y = self.sim(x0, u, noise=True)
        Nt = np.size(u, 0)
        t = np.linspace(0.0, (Nt - 1)* self.__dt, Nt )
        numrows = int(np.ceil(self.__Nx / numcols))

        fig_x = plt.figure()
        for i in range(self.__Nx):
            ax = fig_x.add_subplot(numrows, numcols, i + 1)
            ax.plot(t, y[:, i], 'b-', marker='.', linewidth=1.0)
            ax.set_ylabel('x_' + str(i + 1))
            ax.set_xlabel('Time')
        fig_x.canvas.set_window_title('Model simulation')
        plt.show()


    def predict_compare(self, x0, u, num_cols=2, xnames=None, title=None,):
        """ Predict and compare dicrete RK4 model and linearized model against
            the exact model.
        """
        # Predict future
        Nx = self.__Nx

        dt = self.sampling_time()
        Nt = np.size(u, 0)
        sim_time = Nt * dt

        # Exact model with no noise
        y_exact = self.sim(x0, u, noise=False)
        y_exact = np.vstack([x0, y_exact])

        # RK4
        y_rk4 = np.zeros((Nt + 1 , Nx))
        y_rk4[0] = x0
        for t in range(Nt):
            y_rk4[t + 1]= np.array(self.rk4(y_rk4[t], u[t-1, :], [])).reshape((Nx,))

        #  Linearized Model of Exact discretization
        Ad, Bd = self.discrete_linearize(x0, u[0])
        y_lin = np.zeros((Nt + 1, Nx))
        y_lin[0] = x0
        for t in range(Nt):
            y_lin[t+1] = Ad @ y_lin[t] + Bd @ u[t]

        #  Linearized Model of RK4 discretization
        Ad, Bd = self.discrete_rk4_linearize(x0, u[0])
        y_rk4_lin = np.zeros((Nt + 1, Nx))
        y_rk4_lin[0] = x0
        for t in range(Nt):
            y_rk4_lin[t+1] = Ad @ y_rk4_lin[t] + Bd @ u[t]

        t = np.linspace(0.0, sim_time, Nt + 1)


        num_rows = int(np.ceil(Nx / num_cols))
        if xnames is None:
            xnames = ['State %d' % (i + 1) for i in range(Nx)]

        fontP = FontProperties()
        fontP.set_size('small')
        fig = plt.figure(figsize=(9.0, 6.0))
        for i in range(Nx):
            ax = fig.add_subplot(num_rows, num_cols, i + 1)
            ax.plot(t, y_exact[:, i], 'b-', label='Exact')
            ax.plot(t, y_rk4[:, i], 'r-', label='RK4')
#            ax.plot(t, y_lin[:, i], 'g--', label='Linearized')
#            ax.plot(t, y_lin[:, i], 'y--', label='Linearized RK4')
            ax.set_ylabel(xnames[i])
            ax.legend(prop=fontP, loc='best')
            ax.set_xlabel('Time [s]')
        if title is not None:
            fig.canvas.set_window_title(title)
        else:
            fig.canvas.set_window_title('Compare approximations of system model')
        plt.tight_layout()
        plt.show()
