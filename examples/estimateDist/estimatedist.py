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

################################################################################
# FUNCTION DEFINITIONS
################################################################################

def Hellinger(A, B):
    """
    A = first reference measure (matrix from 2D marginal output)
    B = second reference measure
    """
    n = len(A) # number of bins per dim (assuming regular grid)
    
    # print np.sqrt(1.0 - sum([ np.sqrt(A[i,j]*B[i,j]) for i in range(n) for j in range(n) ]))
    # A = A*(n**2)
    # B = B*(n**2)
    return np.sqrt( 0.5*sum([ ( np.sqrt(A[i,j]) - np.sqrt(B[i,j]) )**2 for i in range(n) for j in range(n)]))

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
    # print '\t Density Evaluated'
    # print '========================================================'
    # print my_discretization._output_probability_set._values
    # print '========================================================'
    # 
    # print '========================================================'
    # print [my_discretization._input_sample_set._values]
    # print '========================================================'
    # Calculate probabilities
    calculateP.prob(my_discretization)
    print '========================================================'
    print [ my_discretization._input_sample_set._probabilities_local, my_discretization._io_ptr_local]
    print ' estimated marginal: '
    print '========================================================'
    # print '\t Probability Calculated\n'
    return my_discretization
    
def my_model(parameter_samples):
    Q_map = np.array([[1.0, 0.0], [0.0, 1.0], [0.1, 0.9]])
    QoI_samples = np.dot(parameter_samples, np.transpose(Q_map))
    return QoI_samples

def identity_model(parameter_samples):
    Q_map = np.array([[1.0, 0.0], [0.0, 1.0]])
    QoI_samples = np.dot(parameter_samples, np.transpose(Q_map))
    return QoI_samples
################################################################################
################################################################################


# np.random.seed(20)
# 
# dim_input = 2 # dimension of paramater space
# # dim_output = 3 # number of QoI
# 
# num_samples_param_space = 1E4 # this is N
# grid_cells_per_dim = 10 # Discretizing the Data Space using a regular grid on the input space - this is = log_dim (M)
# num_samples_emulate_data_space = (grid_cells_per_dim**dim_input)*100 # 1E4 # this is P (think about how many bins we're discretizing with)
# 

# 
# plotting_on = False
# save_disc = False
# MC_assumption = True # (for your input samples)

def generate_data(num_samples_param_space, grid_cells_per_dim, alpha=1, beta=1, plotting_on = False, save_disc = False, MC_assumption = True ):
    # initialize some variables you might pass as parameters later on.
    dim_input = 2
    num_samples_emulate_data_space = (grid_cells_per_dim**dim_input)*100
    dim_range = [0.0, 1.0]
    num_centers = 10

    
    # Define the sampler that will be used to create the discretization
    # object, which is the fundamental object used by BET to compute
    # solutions to the stochastic inverse problem
    
    sampler = bsam.sampler(my_model)
    eye = bsam.sampler(identity_model)

    # Initialize sample objects and discretizations that we will be using.
    # The partition set is drawn from a regular grid to represent 'possible observations'
    Partition_Set = samp.sample_set(dim_input)
    Partition_Set.set_domain(np.repeat([dim_range], dim_input, axis=0))
    Partition_Set = bsam.regular_sample_set(Partition_Set, num_samples_per_dim = np.repeat(grid_cells_per_dim, dim_input, axis=0))


    # The emulated set is drawn from a given density to represent 'likely observations'
    # TODO add in functionality here to change the distribution - look at dim_range (maybe add 'support_range')
    Emulated_Set = samp.sample_set(dim_input)
    Emulated_Set.set_domain(np.repeat([dim_range], dim_input, axis=0))
    Emulated_Set.set_values(np.array( np.transpose([ np.random.beta(a=alpha, b=beta,
                size=num_samples_emulate_data_space) for i in range(dim_input) ]) ))



    # Create Reference Discretization against which you will compare approximations with N samples
    Reference_Set = samp.sample_set(dim_input)
    Reference_Set.set_domain(np.repeat([dim_range], dim_input, axis=0))
    Reference_Set = bsam.regular_sample_set(Reference_Set, num_samples_per_dim = np.repeat(grid_cells_per_dim, dim_input, axis=0))
    Reference_Discretization = eye.compute_QoI_and_create_discretization(Reference_Set)
    Reference_Discretization._input_sample_set.estimate_volume_mc() # The MC assumption is true.
    Reference_Emulation = eye.compute_QoI_and_create_discretization(Emulated_Set)
    simpleFunP.user_partition_user_distribution(Reference_Discretization, 
                                                Reference_Discretization, 
                                                Reference_Emulation)

    Reference_Discretization._input_sample_set.set_probabilities(Reference_Discretization._output_probability_set._probabilities)
    # print '========================================================'
    # print '========================================================'
    # print Reference_Discretization._input_sample_set._values
    # print '========================================================'
    # print '========================================================'
    
    if save_disc == True:
        samp.save_discretization(Reference_Discretization, file_name="0_(%d,%d)_M%d_Reference_Discretization"%(alpha, beta, grid_cells_per_dim ))


    (bins, ref_marginals2D) = plotP.calculate_2D_marginal_probs(Reference_Discretization._input_sample_set, nbins = grid_cells_per_dim)
    if plotting_on == True:
        plotP.plot_2D_marginal_probs(ref_marginals2D, bins, Reference_Discretization._input_sample_set, 
                filename = "1_(%d,%d)_M%d_Reference_Distribution"%(alpha, beta, grid_cells_per_dim), 
                file_extension = ".png", plot_surface=False)

    # plotD.scatter_2D(Reference_Discretization._input_sample_set, filename='ReferenceInputs')

    # Sample from parameter space
    Input_Samples = samp.sample_set(dim_input)
    Input_Samples.set_domain(np.repeat([dim_range], dim_input, axis=0))
    Input_Samples = bsam.random_sample_set('random', Input_Samples, num_samples = num_samples_param_space)
    # Input_Samples = bsam.regular_sample_set(Input_Samples, num_samples_per_dim = np.repeat(grid_cells_per_dim, dim_input, axis=0))
    Input_Samples.estimate_volume_mc()
    My_Discretization = sampler.compute_QoI_and_create_discretization(Input_Samples)

    # Estimate volumes of Voronoi cells associated with the parameter samples
    # if MC_assumption is False:
    #     My_Discretization._input_sample_set.estimate_volume(n_mc_points=10*num_samples_param_space)
    # else:
    #     My_Discretization._input_sample_set.estimate_volume_mc()
    # print 'Experimental Input Sample Discretization Created'



    # Calculate the gradient vectors at some subset of the samples (num_centers).
    # Here the *normalize* argument is set to *True* because we are using bin_ratio to
    # determine the uncertainty in our data.

    # We will approximate the jacobian at each of the centers
    '''
    center_discretization = grad.calculate_gradients_rbf(My_Discretization,
        num_centers, normalize=True)
    print 'Gradients Computed'
    '''
    # With these gradient vectors, we are now ready to choose an optimal set of
    # QoIs to use in the inverse problem, based on optimal skewness properites of
    # QoI vectors.  This method returns a list of matrices.  Each matrix has output_dim rows
    # the first column representing the average skewness of the Jacobian of Q,
    # and the rest of the columns the corresponding QoI indices.

    '''
    input_samples_center = center_discretization.get_input_sample_set()
    best_sets = cqoi.chooseOptQoIs_large(input_samples_center, measure=False)
    print best_sets
    print 'Best Sets Computed'
    '''

    # At this point we have determined the optimal set of QoIs to use in the inverse
    # problem.  Now we compare the support of the inverse solution using
    # different sets of these QoIs.


    # These two objects are the ones we will use to construct our competing data-spaces.
    # We make copies of them and restrict the output space to just the QoI indices we want 
    # to use for inversion. This is the procedure in 'invert_using'
    Partition_Discretization = sampler.compute_QoI_and_create_discretization(Partition_Set)
    #Partition_Discretization._input_sample_set.estimate_volume_mc() # The MC assumption is true.
    Emulated_Discretization = sampler.compute_QoI_and_create_discretization(Emulated_Set)
    # print 'Reference Sample Discretizations Created, Reference Density Computed'
    print ref_marginals2D
    H = []
    for i in range(2): # Possible sets of QoI to choose
        QoI_indices = [i, i+1] # choose up to input_dim
        # print 'QoI Pair %d'%(i+1)
        
        my_discretization = invert_using(My_Discretization, Partition_Discretization, Emulated_Discretization, QoI_indices)
        if save_disc == True:
            samp.save_discretization(my_discretization, file_name="0_(%d,%d)_M%d_N%d_Estimated_Discretization_q%d"%(alpha, beta, grid_cells_per_dim, num_samples_param_space, i))
        # VISUALIZATION
        # plotD.scatter_2D(Emulated_Set)
        (bins, marginals2D) = plotP.calculate_2D_marginal_probs(my_discretization._input_sample_set, nbins = grid_cells_per_dim)
        print marginals2D
        if plotting_on == True:
            plotP.plot_2D_marginal_probs(marginals2D, bins, my_discretization._input_sample_set, 
                            filename = "2_(%d,%d)_M%d_N%d_Recovered_Distribution_q(%d,%d)"%(alpha, beta, grid_cells_per_dim, num_samples_param_space, QoI_indices[0], QoI_indices[1]), 
                            file_extension = ".png", plot_surface=False)

        # marginals2D[(0,1)] yields a matrix of values.
        H.append(Hellinger(marginals2D[(0,1)], ref_marginals2D[(0,1)]))
        # 
        # percentile = 1.0
        # # Sort samples by highest probability density and find how many samples lie in
        # # the support of the inverse solution.  With the Monte Carlo assumption, this
        # # also tells us the approximate volume of this support.
        # (num_samples_in_inverse, _, indices_in_inverse) =\
        #     postTools.sample_highest_prob(top_percentile=percentile,
        #     sample_set=my_discretization._input_sample_set, sort=True)
        # print 'Post-Processing Done'
        # # Print the number of samples that make up the highest percentile percent
        # # samples and ratio of the volume of the parameter domain they take up
        # print (num_samples_in_inverse, np.sum(my_discretization._input_sample_set.get_volumes()[indices_in_inverse]))
    return H