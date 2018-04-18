# -*- coding: utf-8 -*-
# Test script
import numpy as np

n = 3
M = np.eye(n)
mu = np.array([1,1,1])*0
ref = np.array([0,0,0])
x = mu - ref
sigma = np.eye(n)

I = np.eye(n)
S = M @ np.linalg.inv(I + 2*sigma @ M)
c = 1 - np.linalg.det(I + 2*sigma @ M)**(-1/2) * np.exp(- x.T @ S @ x)
print(c)