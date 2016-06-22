# Copyright (C) 2014-2016 The BET Development Team

import numpy as np
import bet.postProcess as postProcess
import bet.calculateP.simpleFunP as simpleFunP
import bet.calculateP.calculateP as calculateP
import bet.postProcess.plotP as plotP
import bet.postProcess.plotDomains as plotD
import bet.sample as samp
import bet.sampling.basicSampling as bsam

np.random.seed(20)

dim_input = 2 # dimension of paramater space
dim_output = 3 # number of QoI

num_samples_emulate_data_space = 1E4 # this is P 
num_samples_param_space = 1E4 # this is N
grid_cells_per_dim = 10 # discretizing the data space using a regular grid on the input space - this is = log_dim (M)
dim_range = [0.0, 1.0]
alpha = 1
beta = 1

# Define the sampler that will be used to create the discretization
# object, which is the fundamental object used by BET to compute
# solutions to the stochastic inverse problem
def my_model(parameter_samples):
    Q_map = np.array([[1.0, 0.0], [0.0, 1.0]])
    QoI_samples = np.dot(parameter_samples, Q_map)
    return QoI_samples

sampler = bsam.sampler(my_model)

# Initialize sample objects and discretizations that we will be using. 
# The partition set is drawn from a regular grid to represent 'possible observations'
Partition_Set = samp.sample_set(dim_input)
Partition_Set.set_domain(np.repeat([dim_range], dim_input, axis=0))
# Sample from regular grid on parameter space. The MC assumption is true.
Partition_Discretization = sampler.create_regular_discretization(Partition_Set,
            num_samples_per_dim = np.repeat(grid_cells_per_dim-1, dim_input, axis=0))
Partition_Discretization._input_sample_set.estimate_volume_mc()

# The emulated set is drawn from a given density to represent 'likely observations'
Emulated_Set = samp.sample_set(dim_input)
Emulated_Set.set_domain(np.repeat([dim_range], dim_input, axis=0))

# add in functionality here to change the distribution - look at dim_range (maybe add 'support_range')
Emulated_Set.set_values(np.array( np.transpose([ np.random.beta(a=alpha, b=beta,
            size=num_samples_emulate_data_space) for i in range(dim_input) ]) ))
Emulated_Set.estimate_volume(n_mc_points=num_samples_emulate_data_space*100)
Emulated_Discretization = sampler.compute_QoI_and_create_discretization(Emulated_Set)


input_samples = samp.sample_set(dim_input)
input_samples.set_domain(np.repeat([dim_range], dim_input, axis=0))






# Sample from parameter space
input_samples = sampler.random_sample_set('random', input_samples,
                num_samples = num_samples_param_space)

MC_assumption = True
# Estimate volumes of Voronoi cells associated with the parameter samples
if MC_assumption is False:
    input_samples.estimate_volume(n_mc_points=1E5)
else:
    input_samples.estimate_volume_mc()



# Create the discretization object using the input samples
my_discretization = sampler.compute_QoI_and_create_discretization(ref_input_samples)
