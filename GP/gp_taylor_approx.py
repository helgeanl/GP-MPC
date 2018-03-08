# -*- coding: utf-8 -*-
"""
Created on Thu Mar 08 14:11:20 2018

@author: helgeanl
"""

# Function definitions for Gaussian Process Model Predictive Control
from sys import path
path.append(r"C:\Users\helgeanl\Google Drive\NTNU\Masteroppgave\casadi-py27-v3.3.0")

#from scipy.stats import norm as norms
#from pylab import *
import numpy as np
#import matplotlib.pyplot as plt
#import math
import casadi as ca


def gp_taylor_approx(invK, X, F, hyper, D, inputmean, inputcov):
    hyper = ca.MX.log(hyper)
    E     = len(invK)
    n     = ca.MX.size(F[:, 1])[0]
    mean  = ca.MX.zeros(E, 1)
    beta  = ca.MX.zeros(n, E)
    log_k = ca.MX.zeros(n, E)
    v     = X - ca.repmat(inputmean, n, 1)

    #invK = MX(invK)
    covariance = ca.MX.zeros(E, E)

    A = ca.SX.sym('A', inputcov.shape)
    [Q, R2] = ca.qr(A)
    determinant = ca.Function('determinant', [A], [ca.exp(ca.trace(ca.log(R2)))])

    for a in range(E):
        ell = hyper[a, :D]
        w = 1/ell**2
        sf2 = ca.MX(hyp[a, D]**2)

        kss = covSEard2(z, z, ell, sf2)
        ks = ca.MX.zeros(n, 1)

        for i in range(n):
            ks[i] = covSEard2(X[i, :] - m, z - m, ell, sf2)
        #ks = repmat()
        ksK = ca.mtimes(ks.T, invK[a])

        mean[a] = ca.mtimes(ksK, Y[:, a] - m) + m
        
        
        
        beta[:, a] = ca.mtimes(invK[a], F[:, a])
        iLambda   = ca.diag(ca.exp(-2 * hyper[a, :D]))
        R  = inputcov + ca.diag(ca.exp(2 * hyper[a, :D]))
        iR = ca.mtimes(iLambda, (ca.MX.eye(D) - ca.solve((ca.MX.eye(D) + ca.mtimes(inputcov, iLambda)), (ca.mtimes(inputcov, iLambda)))))
        T  = ca.mtimes(v, iR)
        c  = ca.exp(2 * hyper[a, D]) / ca.sqrt(determinant(R)) * ca.exp(ca.sum2(hyper[a, :D]))
        q2 = c * ca.exp(-ca.sum2(T * v) * 0.5)
        qb = q2 * beta[:, a]
        mean[a] = ca.sum1(qb)
        t  = ca.repmat(ca.exp(hyper[a, :D]), n, 1)
        v1 = v / t
        log_k[:, a] = 2 * hyper[a, D] - ca.sum2(v1 * v1) * 0.5

    # covariance with noisy input
    for a in range(E):
        ii = v / ca.repmat(ca.exp(2 * hyper[a, :D]), n, 1)
        for b in range(a + 1):
            R = ca.mtimes(inputcov, ca.diag(ca.exp(-2 * hyper[a, :D]) + ca.exp(-2 * hyper[b, :D]))) + ca.MX.eye(D)

            t = 1.0 / ca.sqrt(determinant(R))
            ij = v / ca.repmat(ca.exp(2 * hyper[b, :D]), n, 1)
            Q = ca.exp(ca.repmat(log_k[:, a], 1, n) + ca.repmat(ca.transpose(log_k[:, b]), n, 1) + maha(ii, -ij, ca.solve(R, inputcov * 0.5), n))
            A = ca.mtimes(beta[:, a], ca.transpose(beta[:, b]))
            if b == a:
                A = A - invK[a]
            A = A * Q
            covariance[a, b] = t * ca.sum2(ca.sum1(A))
            covariance[b, a] = covariance[a, b]
        covariance[a, a] = covariance[a, a] + ca.exp(2 * hyper[a, D])
    covariance = covariance - ca.mtimes(mean, ca.transpose(mean))

    return [mean, covariance]


def maha(a1, b1, Q1, n):
    aQ = ca.mtimes(a1, Q1)
    bQ = ca.mtimes(b1, Q1)
    K1  = ca.repmat(ca.sum2(aQ * a1), 1, n) + ca.repmat(ca.transpose(ca.sum2(bQ * b1)), n, 1) - 2 * ca.mtimes(aQ, ca.transpose(b1))
    return K1
