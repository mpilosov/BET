# Copyright (C) 2014-2016 The BET Development Team

import numpy as np
# import bet.calculateP as calculateP
import bet.postProcess as postProcess
import bet.calculateP.simpleFunP as simpleFunP
import bet.calculateP.calculateP as calculateP
import bet.postProcess.plotP as plotP
import bet.postProcess.plotDomains as plotD
import bet.sample as samp
import bet.sampling.basicSampling as bsam

def my_model(parameter_samples):
    Q_map = np.array([[0.506, 0.463],[0.253, 0.918], [0.085, 0.496]])
    QoI_samples = np.dot(parameter_samples,Q_map)
    return QoI_samples

# num_samples_data_space = 1E5 # M
num_samples_emulate_data_space = 1E6
num_samples_param_space = 1E4 # N
dim_input = 2 # dimension of paramater space
dim_output = 3 # number of QoI
ref_grid = np.repeat(25, dim_input, axis=0) # resolution of reference grid
dim_range = [0.0, 1.0]
alpha = 1
beta = 1


# Initialize input parameter sample set object and set domain
ref_input_samples = samp.sample_set(dim_input)
ref_input_samples.set_domain(np.repeat([dim_range], dim_input, axis=0))

emul_input_samples = samp.sample_set(dim_input)
emul_input_samples.set_domain(np.repeat([dim_range], dim_input, axis=0))
# add in functionality here to change domain.
emul_input_samples.set_values(np.array( [ np.random.beta(a=alpha, b=beta,
            size=dim_input) for i in range(int(num_samples_emulate_data_space))]))
# emul_input_samples.update_bounds() # is this needed?

input_samples = samp.sample_set(dim_input)
input_samples.set_domain(np.repeat([dim_range], dim_input, axis=0))

# Define the sampler that will be used to create the discretization
# object, which is the fundamental object used by BET to compute
# solutions to the stochastic inverse problem
sampler = bsam.sampler(my_model)

# Sample from regular grid on parameter space. The MC assumption is true.
ref_input_samples = sampler.regular_sample_set(ref_input_samples,
                num_samples_per_dim = ref_grid)
ref_input_samples.estimate_volume_mc()

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
