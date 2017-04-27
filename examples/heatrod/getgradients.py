import scipy.io as sio
from extractqoivals import filename, temp_locs
import bet.sample as samp
import bet.sensitivity.gradients as bgrad
import bet.sensitivity.chooseQoIs as cqoi
import bet.sampling.basicSampling as bsam
from dolfin import *
import numpy as np
from simulation_2kappas_setup import *
from heatROD import *

matfile = sio.loadmat(filename)

input_samples = samp.sample_set(2)
loaded_input_samples = matfile['samples']
input_samples.set_values(loaded_input_samples)

output_samples = samp.sample_set(len(temp_locs))
loaded_output_samples = matfile['data']
output_samples.set_values(loaded_output_samples)

samp_disc = samp.discretization(input_samples, output_samples)
center_discretization = bgrad.calculate_gradients_rbf(samp_disc, num_centers=100, num_neighbors = 5)
input_samples_centers = center_discretization.get_input_sample_set() # pull center values

index1 = 0
for index2 in range(1+index1,len(temp_locs)):
    (specific_skewness, _) = cqoi.calculate_avg_skewness(input_samples_centers,
            qoi_set=[index1, index2])
    print 'The average skewness of the QoI map defined by indices (%2d, %2d)'%(index1, index2) + \
        ' is ' + str(specific_skewness)
