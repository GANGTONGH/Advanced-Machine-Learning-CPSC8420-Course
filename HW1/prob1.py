#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 14:42:22 2022

@author: gh
"""

import pprint
import numpy as np
from numpy.linalg import inv
from numpy import matmul
import time
import functools
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

def randMat(p,n=10):
    randMat = np.random.rand(n,p)
    return randMat

def lhs(a):
    aT = np.transpose(a)
    lhs = matmul( matmul(aT , a) + 0.1*np.identity(len(a[0])) , aT )
    return lhs

def rhs(a):
    aT = np.transpose(a)
    rhs = matmul( aT , matmul(a , aT) + 0.1*np.identity(len(a)) )
    return rhs

time_l = []
time_r = []

for p in [10,100,1000,2000]:
    a = randMat(p)
    
    time_start = time.perf_counter()
    lhs(a)
    time_elapsed = time.perf_counter() - time_start
    time_l.append(time_elapsed)
    
    time_start = time.perf_counter()
    rhs(a)
    time_elapsed = time.perf_counter() - time_start
    time_r.append(time_elapsed)

x = list(range(4))
xticklabel = [10, 100, 1000, 2000]
plt.scatter(x,time_r)
plt.plot(x,time_r)
plt.scatter(x,time_l)
plt.plot(x,time_l)
plt.xticks(x,labels = xticklabel)
plt.xlabel("p", size = 16)
plt.ylabel("Time (s)", size = 16)
script_dir = os.path.dirname(os.path.realpath(__file__))
print(script_dir)
plt.margins(0)
plt.savefig(script_dir + '/prob1.eps', dpi = 400)
plt.show()
