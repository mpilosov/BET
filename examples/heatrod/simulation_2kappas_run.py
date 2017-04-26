

"""Run the heatplate simulation over a set of samples of the parameter space"""
''' python simulation_2kappas_run test '''

from dolfin import *
import sys, numpy, time
import scipy.io as sio
import random

from simulation_2kappas_setup import *
from heatROD import *

solution_list = []

#loop over each sample and call heatplate
for i in range(len(x)):
    print 'Sample : ', i
    print '(kappa_0, kappa_1) = ', x[i,:]
    kappa_0 = x[i,0]
    kappa_1 = x[i,1]

    heatROD(i, amp, px, width, degree, T_R, kappa_0, kappa_1, rho, cap, nx, mesh, dt, t_stop)

#output parameter and functional data to matlab file
sio.savemat(samples_file_name, {'samples':x})
