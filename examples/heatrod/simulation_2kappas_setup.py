
""" Setup up some model parameters"""

from dolfin import *
import numpy as np
import sys

import bet.sensitivity.gradients as grad
import bet.sample as sample

#time stepping info
dt = 1./100
t_stop = 5.
total_timesteps = t_stop/dt

#Some fixed parameter values
amp = 50.0  #amplitude of the heat source
px = 0  #location of the heat source
py = 0
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
nx = 40
ny = 40
mesh = RectangleMesh(Point(-0.5, -0.5), Point(0.5, 0.5), nx, ny)
parameters['allow_extrapolation'] = True

# set the seed for consistency among data files
np.random.seed(0)

#set points on the plate to approximate the temperature at
num_sensors = 1000
sensor_points = np.random.uniform(-0.5, 0.5, [num_sensors, 2])

# choose sample structure, uniform or cfd
samples_type = 'uniform'

if samples_type == 'uniform':
    #random sampling
    num_samples = 10000
    samples = np.random.uniform(kappa_0_min, kappa_0_max, [num_samples, 2])

elif samples_type == 'cfdclusters':
# cfd samples from BET
    num_centers = 1000

    input_samples = sample.sample_set(2)
    input_samples.set_values(np.random.uniform(kappa_0_min, kappa_0_max, [num_centers, 2]))

    samples = grad.pick_cfd_points(input_samples, radii_vec=0.001)._values

    num_samples = samples.shape[0]







