
""" Setup up some model parameters"""

from dolfin import *
import numpy as np
import sys


#time stepping info
dt = 1./100
t_stop = 1.0
total_timesteps = t_stop/dt

#Some fixed parameter values
amp = 50.0  #amplitude of the heat source
px = 0.5  #location of the heat source
width = 0.05 #width of the heat source
T_R = 0 #initial temp of the plate
cap = 1.5 #heat capacity
rho = 1.5 #density


#here let kappa_0 and kappa_1 be subject to uncertainty
kappa_0_min = .01
kappa_0_max = .2

kappa_1_min = .01
kappa_1_max = .2

#define the mesh properties
degree = 1
nx = 50

mesh = IntervalMesh(nx, 0, 1)
parameters['allow_extrapolation'] = True

#random sampling
np.random.seed(0)
num_samples = 10000
x = (kappa_0_max-kappa_0_min)*np.random.random((num_samples,2)) + kappa_0_min

'''
#Create a uniform grid of the parameter space (kappa_0,kappa_1)
num_divisions = 70
kappa_0_linspace = np.linspace(kappa_0_min, kappa_0_max, num_divisions)
kappa_1_linspace = np.linspace(kappa_1_min, kappa_1_max, num_divisions)

count = 0
x = np.zeros([num_divisions**2,2])
for i in range(len(kappa_0_linspace)):
    for j in range(len(kappa_1_linspace)):
        x[count,:] = [kappa_0_linspace[i], kappa_0_linspace[j]]
        count = count + 1
num_samples = x.shape[0]
'''


