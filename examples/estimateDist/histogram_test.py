
import numpy as np
import bet.postProcess as postProcess
import bet.calculateP.simpleFunP as simpleFunP
import bet.calculateP.calculateP as calculateP
import bet.postProcess.plotP as plotP
import bet.postProcess.plotDomains as plotD
import bet.postProcess.postTools as postTools
import bet.sample as samp
import bet.sampling.basicSampling as bsam
import bet.sensitivity.gradients as grad
import bet.sensitivity.chooseQoIs as cqoi
import bet.Comm as comm
np.random.seed(40)

def identity_model(parameter_samples):
    Q_map = np.array([[1.0, 0.0], [0.0, 1.0]])
    QoI_samples = np.dot(parameter_samples, np.transpose(Q_map))
    return QoI_samples

num_samples_param_space = 40
dim_input = 2
grid_cells_per_dim = 3 # discretization of data space (regular grid)
dim_range = [0.0, 1.0]
alpha = 1
beta = 1
num_samples_emulate_data_space = (grid_cells_per_dim**dim_input)*100

eye = bsam.sampler(identity_model)

# create partition and emulation sets
Partition_Set = samp.sample_set(dim_input)
Partition_Set.set_domain(np.repeat([dim_range], dim_input, axis=0))
Partition_Set = bsam.regular_sample_set(Partition_Set, num_samples_per_dim = np.repeat(grid_cells_per_dim, dim_input, axis=0))

Emulated_Set = samp.sample_set(dim_input)
Emulated_Set.set_domain(np.repeat([dim_range], dim_input, axis=0))
# Emulation from true distribution 
Emulated_Set.set_values(np.array( np.transpose([ np.random.beta(a=alpha, b=beta,
            size=num_samples_emulate_data_space) for i in range(dim_input) ]) ))


Reference_Set = samp.sample_set(dim_input)
Reference_Set.set_domain(np.repeat([dim_range], dim_input, axis=0))
Reference_Set = bsam.regular_sample_set(Reference_Set, num_samples_per_dim = np.repeat(grid_cells_per_dim, dim_input, axis=0))

Reference_Discretization = eye.compute_QoI_and_create_discretization(Reference_Set)
# Reference_Discretization._input_sample_set.estimate_volume_mc() # The MC assumption is true.
Reference_Emulation = eye.compute_QoI_and_create_discretization(Emulated_Set)


simpleFunP.user_partition_user_distribution(Reference_Discretization, 
                                            Reference_Discretization, 
                                            Reference_Emulation)

# set probabilities - the map was the identity, so the inverse problem is trivial
Reference_Discretization._input_sample_set.set_probabilities(Reference_Discretization._output_probability_set._probabilities)

# compute reference marginal
(bins, ref_marginals2D) = plotP.calculate_2D_marginal_probs(Reference_Discretization._input_sample_set, nbins = grid_cells_per_dim)
print 'Reference Marginal:'
print ref_marginals2D[(0,1)]

print '\nReference Probabilities:'
print Reference_Discretization._input_sample_set._probabilities



Input_Samples = samp.sample_set(dim_input)
Input_Samples.set_domain(np.repeat([dim_range], dim_input, axis=0))
Input_Samples = bsam.random_sample_set('random', Input_Samples, num_samples = num_samples_param_space)
# Input_Samples = bsam.regular_sample_set(Input_Samples, num_samples_per_dim = np.repeat(grid_cells_per_dim, dim_input, axis=0))
Input_Samples.estimate_volume_mc()

# Solve Inverse Problem
My_Discretization = eye.compute_QoI_and_create_discretization(Input_Samples)
Partition_Discretization = eye.compute_QoI_and_create_discretization(Partition_Set)
Emulated_Discretization = eye.compute_QoI_and_create_discretization(Emulated_Set)

simpleFunP.user_partition_user_distribution(My_Discretization,
                                            Partition_Discretization,
                                            Emulated_Discretization)
calculateP.prob(My_Discretization)                                            

print '\nInverse Problem:\n Probabilities:'
print My_Discretization._input_sample_set._probabilities_local
print 'Pointer:'
print My_Discretization._io_ptr_local
print '\n Inverse Solution Marginal'
# print marginals for comparison
(bins, marginals2D) = plotP.calculate_2D_marginal_probs(My_Discretization._input_sample_set, nbins = grid_cells_per_dim)
print marginals2D[(0, 1)]
print '\n'