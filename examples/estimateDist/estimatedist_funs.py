import os
import errno
import numpy as np
import bet.sample as samp
import bet.sampling.basicSampling as bsam
import bet.calculateP.simpleFunP as simpleFunP
import bet.calculateP.calculateP as calculateP
from dolfin import * 

def ensure_path_exists(folder):
    try:
        os.makedirs(folder)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def make_model(temp_locs_list):    
    # heatrod code here - including all settings
    t_stop = 1.0

    #Some fixed parameter values
    amp = 50.0  #amplitude of the heat source
    px = 0.5  #location of the heat source
    width = 0.05 #width of the heat source
    T_R = 0 #initial temp of the plate
    cap = 1.5 #heat capacity
    rho = 1.5 #density

    parameters['allow_extrapolation'] = True # 'parameters' reserved for FEnICS

    # mesh properties
    nx = 50 # this is ~ our h value
    mesh = IntervalMesh(nx, 0, 1)
    degree = 1
    r = 1.0 # time stepping ratios - attention to stability
    dt = r/nx
    
    #turn off the heat halfway through
    t_heatoff = t_stop/2.0

    def my_model(parameter_samples):
        QoI_samples = np.zeros( (len(parameter_samples), len(temp_locs_list)) ) 

        for i in range(len(parameter_samples)):
            kappa_0 = parameter_samples[i,0]
            kappa_1 = parameter_samples[i,1]
            
            #split the domain down the middle(dif therm cond)
            kappa_str = 'parameter_samples[0] > 0.5 ?'\
                           'kappa_1 : kappa_0'
                               
            # Physical parameters
            kappa = Expression(kappa_str, kappa_0=kappa_0, kappa_1=kappa_1, degree=1)
            V = FunctionSpace(mesh, 'Lagrange', degree) # define the subspace we will solve the problem in
            # Define initial condition(initial temp of plate)
            T_current = interpolate(Constant(T_R), V)

            # Define variational problem
            T = TrialFunction(V)

            #two f's and L's for heat source on and off
            f_heat = Expression('amp*exp(-(x[0]-px)*(x[0]-px)/width)', amp=amp, px=px, width=width, degree=1)
            f_cool = Constant(0)
            v = TestFunction(V)
            a = rho*cap*T*v*dx + dt*kappa*inner(nabla_grad(v), nabla_grad(T))*dx
            L_heat = (rho*cap*T_current*v + dt*f_heat*v )*dx
            L_cool = (rho*cap*T_current*v + dt*f_cool*v )*dx
            
            A = assemble(a)
            b = None  # variable used for memory savings in assemble calls

            T = Function(V)
            t = dt # initialize first time step
           
            #time stepping method is BWD Euler. (theta = 1) 
            while t <= t_stop:
                if t < t_heatoff:
                    b = assemble(L_heat, tensor=b)
                else:
                    b = assemble(L_cool, tensor=b)
                solve(A, T.vector(), b)

                t += dt
                T_current.assign(T)
               
            # now that the state variable (Temp at time t_stop) has been computed, 
            # we take our point-value measurements
            QoI_samples[i] = np.array([T(xi) for xi in temp_locs_list])
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
    
    return 0.5*(1./num_int_samples)*np.sum( (np.sqrt(den_A) - np.sqrt(den_B) )**2 )
    # THIS RETURNS THE SQUARE OF THE HELLINGER METRIC
    
    # return np.sqrt(0.5*(1./(num_int_samples - B_samples_lost))*np.sum( (np.sqrt(den_A) - np.sqrt(den_B) )**2 ) )
    
    
def invert_using(My_Discretization, Partition_Discretization, Emulated_Discretization, QoI_indices, Emulate = False):
    # Take full discretization objects, a set of indices for the QoI you want 
    # to use to perform inversion, and then do so by redefining the output spaces
    # based on these indices and solving the problem as usual.
    # My_Discretization is copied, the other two are pass-by-reference
    
    # THIS IS FOR THE RECOVERY OF A DISTRIBUTION PROBLEM
    
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

def invert_rect_using(My_Discretization, QoI_indices, Qref, rect_size, cells_per_dimension = 1, Emulate = False):
    # Take full discretization objects, a set of indices for the QoI you want 
    # to use to perform inversion, and then do so by redefining the output spaces
    # based on these indices and solving the problem as usual.
    # My_Discretization is copied, the other two are pass-by-reference
    
    # THIS IS FOR THE INVERTING A REFERENCE POINT PROBLEM
    
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
    
    simpleFunP.regular_partition_uniform_distribution_rectangle_scaled(my_discretization, 
            Q_ref =  Qref[QoI_indices],
            rect_scale = rect_scale )
            
    # simpleFunP.regular_partition_uniform_distribution_rectangle_size(my_discretization, 
    #         Q_ref =  Qref[QoI_indices],
    #         rect_size = rect_size,
    #         cells_per_dimension = cells_per_dimension)
    
    if Emulate:
        # calculateP.prob_on_emulated_samples(my_discretization)
        calculateP.prob_with_emulated_volumes(my_discretization)
    else: 
        calculateP.prob(my_discretization)
        
    return my_discretization
