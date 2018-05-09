# -*- coding: utf-8 -*-
"""
Model Predictive Control with Gaussian Process
@author: Helge-André Langåker
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sys import path
path.append(r"C:\Users\helgeanl\Google Drive\NTNU\Masteroppgave\casadi-py36-v3.4.0")
import time
import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
import casadi.tools as ctools
from scipy.stats import norm
#from matplotlib.font_manager import FontProperties
from .gp_functions import gp_exact_moment, gp_taylor_approx, gp
from .optimize import calc_NLL, train_gp, validate

class MPC:
    def __init__(self, X, Y, meanFunc=None):
        """ Initialize GP model
        """
        