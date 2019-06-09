# MPC with Gaussian Process

A framework for using Gaussian Process together with Model Predictive Control for optimal control.

The framework has been implemented with the principles of being flexible enough to experiment with different GP methods, optimization of GP models. and using different MPC schemes and constraints. Examples provided are obstacle avoidance using an estimated GP model of the vehicle model in (Gao et al., 2011), and level control using a tank system model from (Raff et al., 2006).


The GP methods has been implemented using (Hewig et al., 2017) and (Deisenroth, 2011) as references while the MPC algorithm is a nonlinear stochastic MPC implementation based on (Rawlings et al., 2017), with probabilistic constraints given by (Hewig et al., 2017) . As a backbone in this framework lay CasADi, (Andersson et al., 2018), as a symbolic framework for large scale optimization.

For simulation this framework support the solvers provided by CasADi and Sundails, (Hindmarsh et al., 2005) for both ODEs (CVODES), and DEAs (IDEAS). In addition this framework has implemented a simple Runga-Kutta 4 (RK4) method in CasADi for faster computation of the optimal control problem.

As a model in the MPC algorithm it is possible to use an exact integrator from Sundails (CVODES, IDAS), RK4, GP, a hybrid model consisting of a GP estimating the dynamics and RK4 to integrate the kinematic equation based on the dynamic GP model, or a hybrid where the GP model estimates the noise and modeling error, similar to (Hewig et al., 2017).

This work was developed as a part of the master thesis [Cautious MPC-based control with Machine Learning](https://ntnuopen.ntnu.no/ntnu-xmlui/handle/11250/2572395 "Link to master thesis") (Langåker, 2018).

 


### Requirements
* Python > 3.5
* CasADi (tested with version 3.4)


![alt text](https://github.com/helgeanl/GP-MPC/blob/master/docs/gp.png "Gaussian Process regression")
