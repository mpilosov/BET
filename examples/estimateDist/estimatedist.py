# Copyright (C) 2014-2016 The BET Development Team

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


np.random.seed(20)

dim_input = 2 # dimension of paramater space
# dim_output = 3 # number of QoI

num_samples_emulate_data_space = 1E4 # this is P
num_samples_param_space = 1E5 # this is N
grid_cells_per_dim = 10 # Discretizing the Data Space using a regular grid on the input space - this is = log_dim (M)
dim_range = [0.0, 1.0]
alpha = 1
beta = 5
num_centers = 10

# Define the sampler that will be used to create the discretization
# object, which is the fundamental object used by BET to compute
# solutions to the stochastic inverse problem
def my_model(parameter_samples):
    Q_map = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    QoI_samples = np.dot(parameter_samples, np.transpose(Q_map))
    return QoI_samples

sampler = bsam.sampler(my_model)

# Initialize sample objects and discretizations that we will be using.
# The partition set is drawn from a regular grid to represent 'possible observations'
Partition_Set = samp.sample_set(dim_input)
Partition_Set.set_domain(np.repeat([dim_range], dim_input, axis=0))
Partition_Set = bsam.regular_sample_set(Partition_Set, num_samples_per_dim = np.repeat(grid_cells_per_dim-1, dim_input, axis=0))
Partition_Discretization = sampler.compute_QoI_and_create_discretization(Partition_Set)
Partition_Discretization._input_sample_set.estimate_volume_mc() # The MC assumption is true.
print 'Regular Reference Sample Set Done'

# The emulated set is drawn from a given density to represent 'likely observations'
# TODO add in functionality here to change the distribution - look at dim_range (maybe add 'support_range')
Emulated_Set = samp.sample_set(dim_input)
Emulated_Set.set_domain(np.repeat([dim_range], dim_input, axis=0))
Emulated_Set.set_values(np.array( np.transpose([ np.random.beta(a=alpha, b=beta,
            size=num_samples_emulate_data_space) for i in range(dim_input) ]) ))

# Emulated_Set.set_values(np.array( np.transpose([ np.random.beta(a=alpha, b=beta,
#             size=num_samples_emulate_data_space), np.random.beta(a=beta, b=alpha,
#                         size=num_samples_emulate_data_space) ]) ))
#
Emulated_Discretization = sampler.compute_QoI_and_create_discretization(Emulated_Set)
print 'Emulated Reference Sample Set Done'

# Sample from parameter space
Input_Samples = samp.sample_set(dim_input)
Input_Samples.set_domain(np.repeat([dim_range], dim_input, axis=0))
Input_Samples = bsam.random_sample_set('random', Input_Samples, num_samples = num_samples_param_space)
My_Discretization = sampler.compute_QoI_and_create_discretization(Input_Samples)
MC_assumption = True
# Estimate volumes of Voronoi cells associated with the parameter samples
if MC_assumption is False:
    My_Discretization._input_sample_set.estimate_volume(n_mc_points=10*num_samples_param_space)
else:
    My_Discretization._input_sample_set.estimate_volume_mc()
print 'Experimental Input Sample Set Done'



# Calculate the gradient vectors at some subset of the samples (num_centers).
# Here the *normalize* argument is set to *True* because we are using bin_ratio to
# determine the uncertainty in our data.

# We will approximate the jacobian at each of the centers
center_discretization = grad.calculate_gradients_rbf(My_Discretization,
    num_centers, normalize=True)
print 'Gradients Computed'

# With these gradient vectors, we are now ready to choose an optimal set of
# QoIs to use in the inverse problem, based on optimal skewness properites of
# QoI vectors.  This method returns a list of matrices.  Each matrix has output_dim rows
# the first column representing the average skewness of the Jacobian of Q,
# and the rest of the columns the corresponding QoI indices.
input_samples_center = center_discretization.get_input_sample_set()
best_sets = cqoi.chooseOptQoIs_large(input_samples_center, measure=False)
print 'Best Sets Computed'
# At this point we have determined the optimal set of QoIs to use in the inverse
# problem.  Now we compare the support of the inverse solution using
# different sets of these QoIs.


def invert_using(My_Discretization, Partition_Discretization, Emulated_Discretization, QoI_indices):
    input_samples = My_Discretization._input_sample_set.copy() # might not need to?
    output_samples = My_Discretization._output_sample_set.copy()
    # Choose some QoI indices to solve the ivnerse problem with
    output_samples._dim = len(QoI_indices)
    output_samples.set_values(output_samples.get_values()[:, QoI_indices])

    my_discretization = samp.discretization(input_sample_set=input_samples,
                                            output_sample_set=output_samples)


    partition_output_samples = Partition_Discretization._output_sample_set.copy()
    partition_input_samples = Partition_Discretization._input_sample_set.copy()
    partition_output_samples._dim = len(QoI_indices)
    partition_output_samples.set_values(partition_output_samples.get_values()[:, QoI_indices])
    partition_discretization = samp.discretization(input_sample_set=partition_input_samples,
                                                   output_sample_set=partition_output_samples)

    emulation_output_samples = Emulated_Discretization._output_sample_set.copy()
    emulation_input_samples = Emulated_Discretization._input_sample_set.copy()
    emulation_output_samples._dim = len(QoI_indices)
    emulation_output_samples.set_values(emulation_output_samples.get_values()[:, QoI_indices])
    emulation_discretization = samp.discretization(input_sample_set=emulation_input_samples,
                                                   output_sample_set=emulation_output_samples)

    # Compute the simple function approximation to the distribution on the data space
    simpleFunP.user_partition_user_distribution(my_discretization,
                                                partition_discretization,
                                                emulation_discretization)
    # simpleFunP.user_partition_user_distribution(partition_discretization,
    #                                             partition_discretization,
    #                                             emulation_discretization)
    print 'Density %d Computed'%i
    # Calculate probabilities
    calculateP.prob(my_discretization)
    print 'Probability %d Calculated'%i
    return my_discretization

for i in range(2):

    QoI_indices = [i, i+1] # choose up to input_dim

    my_discretization = invert_using(My_Discretization, Partition_Discretization, Emulated_Discretization, QoI_indices)

    # VISUALIZATION
    # plotD.scatter_2D(Emulated_Set)
    (bins, marginals2D) = plotP.calculate_2D_marginal_probs(my_discretization._input_sample_set, nbins = [20, 20])
    plotP.plot_2D_marginal_probs(marginals2D, bins, my_discretization._input_sample_set, filename = "3_Recovered_Distribution_%d"%i, file_extension = ".png", plot_surface=False)


    percentile = 1.0
    # Sort samples by highest probability density and find how many samples lie in
    # the support of the inverse solution.  With the Monte Carlo assumption, this
    # also tells us the approximate volume of this support.
    (num_samples_in_inverse, _, indices_in_inverse) =\
        postTools.sample_highest_prob(top_percentile=percentile,
        sample_set=my_discretization._input_sample_set, sort=True)
    print 'Post-Processing Done'
    # Print the number of samples that make up the highest percentile percent
    # samples and ratio of the volume of the parameter domain they take up
    print (num_samples_in_inverse, np.sum(my_discretization._input_sample_set.get_volumes()[indices_in_inverse]))
    # This should be the number of samples you used


    my_discretization = invert_using(Partition_Discretization, Partition_Discretization, Emulated_Discretization, QoI_indices)

    # VISUALIZATION
    # plotD.scatter_2D(Emulated_Set)
    (bins, marginals2D) = plotP.calculate_2D_marginal_probs(my_discretization._input_sample_set, nbins = [10, 10])
    plotP.plot_2D_marginal_probs(marginals2D, bins, my_discretization._input_sample_set, filename = "3_Recovered_Distribution_ref_%d"%i, file_extension = ".png", plot_surface=False)


# Hellinger.
#
