import scipy.io as sio
from simulation_2kappas_setup import *
import bet.sample as sample

num_samples = 1
for i in range(num_samples):
    filename = 'Tfiles/Tsample_' + str(i) + '.xml'
    matfile = sio.loadmat(file_name)
    print matfile['samples'][:10]
    
'''
# Initialize the necessary sample objects
input_samples = sample.sample_set(2)
output_samples = sample.sample_set(1000)

# Set the input sample values from the imported file
input_samples.set_values(matfile['samples'])

# Set the data fromthe imported file
output_samples.set_values(matfile['data'])
'''