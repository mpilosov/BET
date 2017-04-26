

"""Run the heatplate simulation over a set of samples of the parameter space"""
''' python simulation_2kappas_run test '''

from dolfin import *
import sys, numpy, time
import scipy.io as sio
import random

from simulation_2kappas_setup import *
from heatplate import *

import time

start_time = time.time()

functional_list = []

# gather data every 50 time steps
data = np.zeros([num_samples, num_sensors * total_timesteps / 50])

#loop over each sample and call heatplate
for i in range(samples.shape[0]):
    print '\nSample : ', i, ' / ', samples.shape[0]
    print '(kappa_0, kappa_1) = ', samples[i,:]
    kappa_0 = samples[i,0]
    kappa_1 = samples[i,1]

    heatplatecenter(i, amp, px, py, width, degree, T_R, kappa_0, kappa_1, rho, cap, nx, ny, mesh, sensor_points, dt, t_stop, data)

    print 'current run_time : ', time.time() - start_time


run_time = time.time() - start_time

print ' '
print 'run_time : ', run_time


#output parameter and functional data to matlab file
filename = './data/fenics_data/heatplate_' + \
                     samples_type + '_' + \
                     str(int(num_samples)) + 'samples_' + \
                     str(int(num_sensors)) + 'sensors_' + \
                     str(int(width)) + 'width_' + \
                     str(int(px)) + 'px_' + \
                     str(int(py)) + 'py_' + \
                     str(int(nx)) + 'nx_' + \
                     str(int(ny)) + 'ny_' + \
                     str(int(t_stop)) + 'tstop'
sio.savemat(filename, {'samples':samples,
                       'data':data,
                       'sensor_points':sensor_points,
                       'amp':amp,
                       'px':px,
                       'py':py,
                       'nx':nx,
                       'ny':ny,
                       'width':width,
                       'T_R':T_R,
                       'cap':cap,
                       'rho':rho,
                       'kappa_0_min':kappa_0_min,
                       'kappa_0_max':kappa_0_min})



