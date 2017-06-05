import scipy.io as sio
from simulation_2kappas_setup import nx, samples_file_name, num_samples
import bet.sample as sample
from dolfin import *
import numpy as np

# IF you change the functionals 
filename = 'functionaldata' # place where samples and functional values are stored.
matfile = sio.loadmat(samples_file_name)
# loaded_input_samples = matfile['samples']

# Extract QoI data from state variable T in folder Tfiles
# temp_locs = [0.1, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95]
# temp_locs = [g/100.0 for g in range(1,100)]
# temp_locs = [0.125, 0.9, 0.7] # list of QoI locations 
temp_locs = [0.25, 0.51, 0.67, 0.98]
# temp_locs = [temp_locs2[i]+.01 for i in range(4)]
# IF you change the functionals, make sure to re-run this script before getgradients

# temp_locs = [0.05, 0.95]
qoi_data = np.zeros( (num_samples, len(temp_locs)) ) # initialize empty array

mesh = IntervalMesh(nx, 0, 1)
V = FunctionSpace(mesh, 'Lagrange', 1)

for i in range(num_samples):
    file_name = 'Tfiles/Tsample_' + str(i) + '.xml'
    T = Function(V, file_name) # Loaded up temperature profile for each sample
    qoi_data[i] = np.array([T(xi) for xi in temp_locs]) # take temperature values at points of interest

sio.savemat(filename, {'samples':matfile['samples'], 'data':qoi_data})


'''
# Initialize the necessary sample objects
input_samples = sample.sample_set(2)
output_samples = sample.sample_set(1000)

# Set the input sample values from the imported file
input_samples.set_values(matfile['samples'])

# Set the data fromthe imported file
output_samples.set_values(matfile['data'])
'''