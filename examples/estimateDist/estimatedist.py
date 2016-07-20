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

def Hellinger(A, B):
    """
    A = first reference measure (matrix from 2D marginal output)
    B = second reference measure
    """
    n = len(A) # number of bins per dim (assuming regular grid)
    # Alternative (less computationally stable?) method of computing distance
    # print np.sqrt(1.0 - sum([ np.sqrt(A[i,j]*B[i,j]) for i in range(n) for j in range(n) ]))
    # A = A*(n**2) # still not entirely convinced these aren't necessary. 
    # B = B*(n**2)
    return np.sqrt( 0.5*sum([ ( np.sqrt(A[i,j]) - np.sqrt(B[i,j]) )**2 for i in range(n) for j in range(n)]))

def invert_using(My_Discretization, Partition_Discretization, Emulated_Discretization, QoI_indices):
    # Choose some QoI indices to solve the ivnerse problem with
    input_samples = My_Discretization._input_sample_set.copy() # might not need to copy?
    output_samples = My_Discretization._output_sample_set.copy()
    
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

    emulated_output_samples = Emulated_Discretization._output_sample_set.copy()
    emulated_input_samples = Emulated_Discretization._input_sample_set.copy()
    emulated_output_samples._dim = len(QoI_indices)
    emulated_output_samples.set_values(emulated_output_samples.get_values()[:, QoI_indices])
    emulated_discretization = samp.discretization(input_sample_set=emulated_input_samples,
                                                   output_sample_set=emulated_output_samples)

    # Compute the simple function approximation to the distribution on the data space of interest
    simpleFunP.user_partition_user_distribution(my_discretization,
                                                partition_discretization,
                                                emulated_discretization)
    print 'about to call prob'                                            
    calculateP.prob(my_discretization)
    
    return my_discretization
    
    
def generate_reference(grid_cells_per_dim, alpha, beta, save_ref_disc = True, save_ref_plot = False):
    dim_input = 2
    dim_range = [0.0, 1.0]
    emulation_constant = 100
    num_samples_emulate_data_space = (grid_cells_per_dim**dim_input)*emulation_constant

    # Create Reference Discretization against which you will compare approximations with N samples
    print '\nComputing Reference Discretization for M = %4d'%(grid_cells_per_dim**2)
    Partition_Set = samp.sample_set(dim_input)
    Partition_Set.set_domain(np.repeat([dim_range], dim_input, axis=0))
    Partition_Set = bsam.regular_sample_set(Partition_Set, num_samples_per_dim = np.repeat(grid_cells_per_dim, dim_input, axis=0) )
    Reference_Discretization = samp.discretization(Partition_Set, Partition_Set)
    
    np.random.seed(grid_cells_per_dim) # This yields consistent results.
    
    Emulated_Set = samp.sample_set(dim_input)
    Emulated_Set.set_domain(np.repeat([[0.0, 1.0]], dim_input, axis=0))
    # Emulated_Set = bsam.regular_sample_set(Emulated_Set, num_samples_per_dim = 3*np.repeat(grid_cells_per_dim, dim_input, axis=0))
    Emulated_Set.set_values(np.array( np.transpose([ np.random.beta(a=alpha, b=beta,
                size=num_samples_emulate_data_space) for i in range(dim_input) ]) ))
    
    Reference_Discretization._input_sample_set.estimate_volume_mc() # The MC assumption is true.
    Reference_Emulation_Discretization = samp.discretization(Emulated_Set, Emulated_Set)
    simpleFunP.user_partition_user_distribution(Reference_Discretization, 
                                                Reference_Discretization, 
                                                Reference_Emulation_Discretization)

    Reference_Discretization._input_sample_set.set_probabilities(Reference_Discretization._output_probability_set._probabilities)
    if save_ref_disc == True:
        samp.save_discretization(Reference_Discretization, file_name="0_(%d,%d)_M%d_Reference_Discretization"%(alpha, beta, grid_cells_per_dim ))
    if save_ref_plot == True:
        (bins, ref_marginal) = plotP.calculate_2D_marginal_probs(Reference_Discretization._input_sample_set, nbins = grid_cells_per_dim)
        plotP.plot_2D_marginal_probs(ref_marginal, bins, Reference_Discretization._input_sample_set, 
                    filename = "1_(%d,%d)_M%d_Reference_Distribution"%(alpha, beta, grid_cells_per_dim), 
                    file_extension = ".png", plot_surface=False)
    return Reference_Discretization, Partition_Set, Emulated_Set

def generate_N_reference(my_model, N_grid_cells_per_dim, M_grid_cells_per_dim, alpha, beta, save_ref_disc = True, save_ref_plot = False):
    _, Partition_Set, Emulated_Set = generate_reference(M_grid_cells_per_dim, alpha, beta, save_ref_disc, save_ref_plot)
    
    
    # initialize some variables you might pass as parameters later on.
    dim_input = 2 # definitely can pull this from partition_set
    dim_range = [0.0, 1.0] # probably can pull this from partition_set 
    
    # Define the sampler that will be used to create the discretization object - 2D for now only.
    sampler = bsam.sampler(my_model)
    # np.random.seed(num_samples_param_space)
    
    # Sample from parameter space
    Input_Samples = samp.sample_set(dim_input)
    Input_Samples.set_domain(np.repeat([dim_range], dim_input, axis=0))
    Input_Samples = bsam.regular_sample_set(Input_Samples, num_samples_per_dim = np.repeat(N_grid_cells_per_dim, dim_input, axis=0) )
    Input_Samples.estimate_volume_mc()
    
    My_Discretization = sampler.compute_QoI_and_create_discretization(Input_Samples)
    Partition_Discretization = sampler.compute_QoI_and_create_discretization(Partition_Set)
    Emulated_Discretization = sampler.compute_QoI_and_create_discretization(Emulated_Set)
    QoI_indices = [0, 1]
    Reference_Discretization = invert_using(My_Discretization, Partition_Discretization, Emulated_Discretization, QoI_indices)
    
    return Reference_Discretization, Partition_Set, Emulated_Set 

def generate_model_discretizations(my_model, Partition_Set, Emulated_Set, num_samples_param_space, alpha=1, beta=1):
    # initialize some variables you might pass as parameters later on.
    dim_input = 2 # definitely can pull this from partition_set
    dim_range = [0.0, 1.0] # probably can pull this from partition_set 
    
    # Define the sampler that will be used to create the discretization object - 2D for now only.
    sampler = bsam.sampler(my_model)
    # np.random.seed(num_samples_param_space)
    
    # Sample from parameter space
    Input_Samples = samp.sample_set(dim_input)
    Input_Samples.set_domain(np.repeat([dim_range], dim_input, axis=0))
    Input_Samples = bsam.random_sample_set('random', Input_Samples, num_samples = num_samples_param_space)
    # Input_Samples = bsam.regular_sample_set(Input_Samples, num_samples_per_dim = np.repeat(np.int(np.sqrt(num_samples_param_space)), dim_input, axis=0) )
    Input_Samples.estimate_volume_mc()
    # Input_Samples.estimate_volume(n_mc_points=100*num_samples_param_space)
    
    My_Discretization = sampler.compute_QoI_and_create_discretization(Input_Samples)
    Partition_Discretization = sampler.compute_QoI_and_create_discretization(Partition_Set)
    Emulated_Discretization = sampler.compute_QoI_and_create_discretization(Emulated_Set)
    
    return My_Discretization, Partition_Discretization, Emulated_Discretization
