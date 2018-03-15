# -*- coding: utf-8 -*-
"""
# Copyright (c) 2018
@author: Helge-André Langåker
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sys import path
path.append(r"C:\Users\helgeanl\Google Drive\NTNU\Masteroppgave\casadi-py27-v3.3.0")

import numpy as np


def covSEard(x, z, ell, sf2):
    """ GP squared exponential kernel """
    #dist = 0
    #for i in range(len(x)):
    #    dist = dist + (x[i] - z[i])**2 / (ell[i]**2)
    dist = np.sum((x - z)**2 / ell**2)
    return sf2 * np.exp(-.5 * dist)


def calc_cov_matrix(X, ell, sf2):
    """ GP squared exponential kernel """
    dist = 0
    n, D = X.shape
    for i in range(D):
        x = X[:, i].reshape(n, 1)
        dist = (np.sum(x**2, 1).reshape(-1, 1) + np.sum(x**2, 1) -
                2 * np.dot(x, x.T)) / ell[i]**2 + dist
    return sf2 * np.exp(-.5 * dist)


def gp(hyp, invK, X, Y, u):
    n, D = X.shape
    ell = hyp[:D]
    sf2 = hyp[D]**2
    #m   = hyp[D + 2]
    kss = covSEard(u, u, ell, sf2)
    ks = np.zeros(n)
    for i in range(n):
        ks[i] = covSEard(X[i, :], u, ell, sf2)
    ksK = np.dot(ks.T, invK)
    mu = np.dot(ksK, Y)
    s2 = kss - np.dot(ksK, ks)
    return mu, s2
