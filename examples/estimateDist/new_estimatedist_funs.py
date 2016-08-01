
import os
import errno
import numpy as np
import bet.sample as samp
import bet.sampling.basicSampling as bsam
import bet.calculateP.simpleFunP as simpleFunP
import bet.calculateP.calculateP as calculateP

def ensure_path_exists(folder):
    try:
        os.makedirs(folder)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
def make_model(skew_range):
    # this function makes a linear map whos first component is the x-unit vector
    # and each subsequent component is a norm-1 vector satisfying the property
    # that the 2-2 map made from it and the aforementioned unit vector is a map
    # with skewness in skew_range, which is a list of desired skewnesses   
    # TODO currently this map only works for 2-D input space     
    
    def my_model(parameter_samples):
        Q_map = [ [1.0, 0.0] ] # all map components have the same norm, rect_size to have measures of events equal btwn spaces.
        for s in skew_range:
            theta = np.arcsin(1./s)
            # Q_map.append( [np.cos(theta), np.sin(theta)] ) # all map components have the same norm
            # below with rect_size to have measures of events equal between spaces.
            Q_map.append( [s*np.cos(theta), s*np.sin(theta)] ) 
        Q_map = np.array ( Q_map )
        QoI_samples = np.dot(parameter_samples, np.transpose(Q_map))
        return QoI_samples
    return my_model

def mc_Hellinger(integration_sample_set, set_A, set_A_ptr, set_B, set_B_ptr ):
    # Aset, Bset are sample_set type objects, to be evaluated at the points in 
    # integration_sample_set. the pointers are from integration set into the others
    if isinstance(set_A, samp.discretization):
        set_A = set_A.get_input_sample_set()
    if isinstance(set_B, samp.discretization):
        set_B = set_B.get_input_sample_set()
    num_int_samples = integration_sample_set.check_num()
    
    A_prob = set_A._probabilities[set_A_ptr]
    A_vol = set_A._volumes[set_A_ptr]
    
    B_prob = set_B._probabilities[set_B_ptr]
    B_vol = set_B._volumes[set_B_ptr]
    
    # prevents divide by zero. adheres to convention that if a cell has 
    # zero volume due to inadequate emulation, then it will get assigned zero probability as well
    A_samples_lost = len(A_vol[A_vol == 0])
    B_samples_lost = len(B_vol[B_vol == 0])
    A_vol[A_vol == 0] = 1
    B_vol[B_vol == 0] = 1
    
    den_A = np.divide( A_prob, A_vol)
    den_B = np.divide( B_prob, B_vol )
    
    # B_prob[A_prob == 0] = 0
    if B_samples_lost>0:
        print '\t samples lost = %4d'%(B_samples_lost)
    if A_samples_lost>0:
        print '\t !!!!! Integration samples lost = %4d'%(A_samples_lost)
    
    # return 0.5*(1./num_int_samples)*np.sum( (np.sqrt(den_A) - np.sqrt(den_B) )**2 )
    return 0.5*(1./(num_int_samples - B_samples_lost))*np.sum( (np.sqrt(den_A) - np.sqrt(den_B) )**2 )
    
    
def invert_using(My_Discretization, Partition_Discretization, Emulated_Discretization, QoI_indices, Emulate = False):
    # Take full discretization objects, a set of indices for the QoI you want 
    # to use to perform inversion, and then do so by redefining the output spaces
    # based on these indices and solving the problem as usual.
    # My_Discretization is copied, the other two are pass-by-reference
    input_samples = My_Discretization._input_sample_set.copy()
    output_samples = My_Discretization._output_sample_set.copy()
    
    output_samples._dim = len(QoI_indices)
    output_samples.set_values(output_samples._values[:, QoI_indices])
    
    partition_output_samples = Partition_Discretization._output_sample_set.copy()
    partition_output_samples._dim = len(QoI_indices)
    partition_output_samples.set_values(partition_output_samples._values[:, QoI_indices])
    
    emulated_output_samples = Emulated_Discretization._output_sample_set.copy()
    emulated_output_samples._dim = len(QoI_indices)
    emulated_output_samples.set_values(emulated_output_samples._values[:, QoI_indices])
    
    output_samples.global_to_local()
    partition_output_samples.global_to_local()
    emulated_output_samples.global_to_local()
    partition_discretization = samp.discretization(input_sample_set=Partition_Discretization._input_sample_set,
                                                   output_sample_set=partition_output_samples)
    emulated_discretization = samp.discretization(input_sample_set=Emulated_Discretization._input_sample_set,
                                                   output_sample_set=emulated_output_samples)
    my_discretization = samp.discretization(input_sample_set=input_samples,
                                            output_sample_set=output_samples)
    if Emulate:
        my_discretization.set_emulated_input_sample_set(My_Discretization._emulated_input_sample_set.copy() )
    
    
    # Compute the simple function approximation to the distribution on the data space of interest
    simpleFunP.user_partition_user_distribution(my_discretization,
                                                partition_discretization,
                                                emulated_discretization)
    
    if Emulate:
        # calculateP.prob_on_emulated_samples(my_discretization)
        calculateP.prob_with_emulated_volumes(my_discretization)
    else: 
        calculateP.prob(my_discretization)
        
    return my_discretization

def invert_rect_using(My_Discretization, QoI_indices, Qref, rect_scale, Emulate = False):
    # Take full discretization objects, a set of indices for the QoI you want 
    # to use to perform inversion, and then do so by redefining the output spaces
    # based on these indices and solving the problem as usual.
    # My_Discretization is copied, the other two are pass-by-reference
    input_samples = My_Discretization._input_sample_set.copy()
    output_samples = My_Discretization._output_sample_set.copy()
    
    output_samples._dim = len(QoI_indices)
    output_samples.set_values(output_samples._values[:, QoI_indices])
    
    output_samples.global_to_local()

    my_discretization = samp.discretization(input_sample_set=input_samples,
                                            output_sample_set=output_samples)
    if Emulate:
        my_discretization.set_emulated_input_sample_set(My_Discretization._emulated_input_sample_set.copy() )
    
    # Compute the simple function approximation to the distribution on the data space of interest
    
    # simpleFunP.regular_partition_uniform_distribution_rectangle_scaled(my_discretization, 
    #         Q_ref =  Qref[QoI_indices],
    #         rect_scale = rect_scale )
            
    simpleFunP.regular_partition_uniform_distribution_rectangle_size(my_discretization, 
            Q_ref =  Qref[QoI_indices],
            rect_size= rect_scale )
    
    if Emulate:
        # calculateP.prob_on_emulated_samples(my_discretization)
        calculateP.prob_with_emulated_volumes(my_discretization)
    else: 
        calculateP.prob(my_discretization)
        
    return my_discretization
