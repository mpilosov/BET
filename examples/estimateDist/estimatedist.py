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
    input_samples = My_Discretization._input_sample_set#.copy() # might not need to copy?
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

    calculateP.prob(my_discretization)
    return my_discretization
    
def my_model(parameter_samples):
    Q_map = np.array([[1.0, 1.0], [1.0, -1.0], [1.0, 0.0]])
    QoI_samples = np.dot(parameter_samples, np.transpose(Q_map))
    return QoI_samples

def generate_reference(dim_input, grid_cells_per_dim, alpha, beta, save_disc = True, save_plot = False):
    # Create Reference Discretization against which you will compare approximations with N samples
    Reference_Set = samp.sample_set(dim_input)
    Reference_Set.set_domain(np.repeat([dim_range], dim_input, axis=0))
    Reference_Set = bsam.regular_sample_set(Reference_Set, num_samples_per_dim = np.repeat(grid_cells_per_dim, dim_input, axis=0))
    Reference_Discretization = samp.discretization(Reference_Set, Reference_Set)
    
    num_samples_emulate_data_space = grid_cells_per_dim**dim_input*100;
    
    Emulated_Set = samp.sample_set(dim_input)
    Emulated_Set.set_domain(np.repeat([[0.0, 1.0]], dim_input, axis=0))
    Emulated_Set.set_values(np.array( np.transpose([ np.random.beta(a=alpha, b=beta,
                size=num_samples_emulate_data_space) for i in range(dim_input) ]) ))

    # Reference_Discretization._input_sample_set.estimate_volume_mc() # The MC assumption is true.
    Reference_Emulation = eye.compute_QoI_and_create_discretization(Emulated_Set)
    simpleFunP.user_partition_user_distribution(Reference_Discretization, 
                                                Reference_Discretization, 
                                                Reference_Emulation)

    Reference_Discretization._input_sample_set.set_probabilities(Reference_Discretization._output_probability_set._probabilities)
    if save_disc == True:
        samp.save_discretization(Reference_Discretization, file_name="0_(%d,%d)_M%d_Reference_Discretization"%(alpha, beta, grid_cells_per_dim ))
    
    (bins, ref_marginals2D) = plotP.calculate_2D_marginal_probs(Reference_Discretization._input_sample_set, nbins = grid_cells_per_dim)
    if save_plot == True:
        plotP.plot_2D_marginal_probs(ref_marginals2D, bins, Reference_Discretization._input_sample_set, 
                filename = "1_(%d,%d)_M%d_Reference_Distribution"%(alpha, beta, grid_cells_per_dim), 
                file_extension = ".png", plot_surface=False)
    return


def generate_data(num_samples_param_space, grid_cells_per_dim, alpha=1, beta=1, plotting_on = False, save_disc = False, MC_assumption = True ):
    # initialize some variables you might pass as parameters later on.
    dim_input = 2
    num_samples_emulate_data_space = (grid_cells_per_dim**dim_input)*100
    dim_range = [0.0, 1.0]
    num_centers = 10
    
    # Define the sampler that will be used to create the discretization object
    sampler = bsam.sampler(my_model)

    # Initialize sample objects and discretizations that we will be using.
    # The partition set is drawn from a regular grid to represent 'possible observations'
    Partition_Set = samp.sample_set(dim_input)
    Partition_Set.set_domain(np.repeat([dim_range], dim_input, axis=0))
    Partition_Set = bsam.regular_sample_set(Partition_Set, num_samples_per_dim = np.repeat(grid_cells_per_dim, dim_input, axis=0))


    # The emulated set is drawn from a given density to represent 'likely observations'
    # TODO add in functionality here to change the distribution - look at dim_range (maybe add 'support_range')
    Emulated_Set = samp.sample_set(dim_input)
    Emulated_Set.set_domain(np.repeat([[0.0, 1.0]], dim_input, axis=0))
    Emulated_Set.set_values(np.array( np.transpose([ np.random.beta(a=alpha, b=beta,
                size=num_samples_emulate_data_space) for i in range(dim_input) ]) ))


    # Sample from parameter space
    Input_Samples = samp.sample_set(dim_input)
    Input_Samples.set_domain(np.repeat([dim_range], dim_input, axis=0))
    Input_Samples = bsam.random_sample_set('random', Input_Samples, num_samples = num_samples_param_space)
    Input_Samples.estimate_volume_mc()
    # Input_Samples.estimate_volume(n_mc_points=100*num_samples_param_space)
    My_Discretization = sampler.compute_QoI_and_create_discretization(Input_Samples)

    # These two objects are the ones we will use to construct our competing data-spaces.
    # We make copies of them and restrict the output space to just the QoI indices we want 
    # to use for inversion. This is the procedure in 'invert_using'
    Partition_Discretization = sampler.compute_QoI_and_create_discretization(Partition_Set)
    #Partition_Discretization._input_sample_set.estimate_volume_mc() # The MC assumption is true.
    Emulated_Discretization = sampler.compute_QoI_and_create_discretization(Emulated_Set)

    
    H = [] # vector to store Hellinger Distances.
    # TODO enumerate all possible choices of QoI maps. -- or change my_model each time? 
    for i in range(2): # Possible sets of QoI to choose
        QoI_indices = [i, i+1] # choose up to input_dim
        # print 'QoI Pair %d'%(i+1)
        
        my_discretization = invert_using(My_Discretization, Partition_Discretization, Emulated_Discretization, QoI_indices)
        
        if save_disc == True:
            samp.save_discretization(my_discretization, file_name="0_(%d,%d)_M%d_N%d_Estimated_Discretization_q%d"%(alpha, beta, grid_cells_per_dim, num_samples_param_space, i))

        (bins, marginals2D) = plotP.calculate_2D_marginal_probs(my_discretization._input_sample_set, nbins = grid_cells_per_dim)
        
        if plotting_on == True & i == 0:
            plotP.plot_2D_marginal_probs(marginals2D, bins, my_discretization._input_sample_set, 
                            filename = "2_(%d,%d)_M%d_N%d_Recovered_Distribution_q(%d,%d)"%(alpha, beta, grid_cells_per_dim, num_samples_param_space, QoI_indices[0], QoI_indices[1]), 
                            file_extension = ".png", plot_surface=False)

        # marginals2D[(0,1)] yields a matrix of values.
        H.append(Hellinger(marginals2D[(0,1)], ref_marginals2D[(0,1)]))

    return H