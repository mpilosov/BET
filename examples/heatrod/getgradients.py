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
from sys import *

matfile = sio.loadmat(filename)

input_samples = samp.sample_set(2)
loaded_input_samples = matfile['samples']
input_samples.set_values(loaded_input_samples)

output_samples = samp.sample_set(len(temp_locs))
loaded_output_samples = matfile['data']
output_samples.set_values(loaded_output_samples)

samp_disc = samp.discretization(input_samples, output_samples)
center_discretization = bgrad.calculate_gradients_rbf(samp_disc, num_centers=200, num_neighbors = 5)
# full and reference (single)
input_samples_centers = center_discretization.get_input_sample_set() # pull center values
min_skew = 3
max_skew = 1
index1 = 0
# for index1 in range(0,len(temp_locs)):
    # for index2 in range(1+index1,len(temp_locs)):
#         (specific_skewness, _) = cqoi.calculate_avg_skewness(input_samples_centers, qoi_set=[index1, index2])
#         (specific_scaling, _) = cqoi.calculate_avg_measure(input_samples_centers, qoi_set=[index1, index2])
#         if np.abs(specific_scaling-float(sys.argv[1]) ) < 0.05:
#             if specific_skewness < min_skew:
#                 min_skew = specific_skewness
#                 idx_min = [index1, index2]
#             if specific_skewness > max_skew:
#                 max_skew = specific_skewness
#                 idx_max = [index1, index2]
# print 'skew range = %2.2f - %2.2f'%(min_skew, max_skew)
# print 'at indices [%d, %d] and [%d, %d]'%(idx_min[0], idx_min[1], idx_max[0], idx_max[1]) + '\n'
for index1 in range(0,len(temp_locs)):
    for index2 in range(1+index1,len(temp_locs)):
        (specific_skewness, _) = cqoi.calculate_avg_skewness(input_samples_centers, qoi_set=[index1, index2])
        (specific_scaling, _) = cqoi.calculate_avg_measure(input_samples_centers, qoi_set=[index1, index2])
        
        print 'The average (skew, meas) of the QoI_(%1d, %1d) = (%2.2f, %2.2f)'%(index1, index2, specific_skewness, specific_scaling)
