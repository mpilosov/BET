from new_estimatedist_funs import *
from new_estimatedist_setup import *


integration_sets_filenames = []
ref_solutions_filenames = []
est_solutions_filenames = []

ref_sol_int_ptrs_filenames = []
est_sol_int_ptrs_filenames = []

est_discretizations_filenames = [] # dont think I need this one

emul_filename = ref_sol_dir + 'Emulation_Discretization'

# Create Integration Sets
for I in I_values: # resolution of integration mesh
    Ival = I**(1 + (dim_input-1)*(integration_mesh_type == 'reg') )
    temp_dir = cwd +  '/' + results_dir + '/' + sub_dirs[1] + '/'
    ensure_path_exists(temp_dir)
    filename = temp_dir + '%s_IntSet_%d'%(integration_mesh_type, Ival)
    integration_sets_filenames.append(filename)
    
    integration_sample_set = samp.sample_set(dim_input)
    integration_sample_set.set_domain(dim_range)
    if integration_mesh_type == 'reg': # regular mesh for monte-carlo integration
        
        # NOTE: not sure if this provides any speedup because this is the input set 
        # (into the computationally expensive KDTree being queried) is the reference mesh.
        # speedup happens with the reference mesh being a cartesian_sample_set.
        
        # integration_sample_set = samp.cartesian_sample_set(dim_input)
        # integration_sample_set.set_domain(dim_range)
        # lin_mesh = []
        # for each_dim_range in dim_range:
        #     lin_mesh.append( np.linspace(each_dim_range[0], each_dim_range[1], I+1) )
        # integration_sample_set.setup(lin_mesh)
            
        integration_sample_set = bsam.regular_sample_set(integration_sample_set, num_samples_per_dim = np.repeat(I, dim_input, axis=0))
    else: # random mesh for monte-carlo integration
        integration_sample_set = bsam.random_sample_set('random', 
                integration_sample_set, num_samples = I)
    integration_sample_set.estimate_volume_mc()
    samp.save_sample_set(integration_sample_set, filename)
print 'Integration Sets Computed'    


# Create Reference Discretizations and Pointers to Integration Sets
for BigN in BigN_values:
    BigNval = BigN**(1 + (dim_input-1)*(reference_mesh_type == 'reg') )
    ref_sol_dir_2 = ref_sol_dir + '%s_BigN_%d'%(reference_mesh_type, BigNval) + '/'
    ensure_path_exists(ref_sol_dir_2)
    
    ref_sample_set = samp.sample_set(dim_input)
    ref_sample_set.set_domain(dim_range)
    if reference_mesh_type == 'reg':
        
        # NOTE: yields warnings but no errors apparent - definitely much faster
        # ref_sample_set = samp.cartesian_sample_set(dim_input)
        # ref_sample_set.set_domain(dim_range)
        # lin_mesh = []
        # for each_dim_range in dim_range:
        #     lin_mesh.append( np.linspace(each_dim_range[0], each_dim_range[1], BigN+1) )
        # ref_sample_set.setup(lin_mesh)
        
        ref_sample_set =  bsam.regular_sample_set(ref_sample_set, num_samples_per_dim = np.repeat(BigN, dim_input, axis=0))
    else:
        ref_sample_set = bsam.random_sample_set('random', 
                ref_sample_set, num_samples = BigN)
    ref_sample_set.estimate_volume_mc()
    Ref_Discretization = sampler.compute_QoI_and_create_discretization(ref_sample_set)
    ref_filename = 'Reference_Disc-' + '%s_BigN_%d'%(reference_mesh_type, BigNval) 
    samp.save_discretization(Ref_Discretization, ref_sol_dir_2 + ref_filename )
    
    # Generate io_pointer for integration sets
    for int_mesh_num in range(len(I_values)):
        Ival = I_values[int_mesh_num]**(1 + (dim_input-1)*(integration_mesh_type == 'reg') )
        filename =ref_sol_dir_2 + 'ptr_from_' + \
                '%s_I_%d'%(integration_mesh_type, Ival) + '_to_' + \
                 '%s_BigN_%d'%(reference_mesh_type, BigNval) 
                
        int_mesh_set = samp.load_sample_set(integration_sets_filenames[int_mesh_num])  
        (_, ptr) = ref_sample_set.query(int_mesh_set._values)
        np.save(filename, ptr)
        ref_sol_int_ptrs_filenames.append( filename + '.npy')
    # print ref_sol_int_ptrs_filenames
print 'Reference Discretizations Computed, Pointers from Integration Sets Created'

# Create one emulation set instead of one for each M. 
Emulation_Set = samp.sample_set(dim_input) # generate emulated set from true distribution
Emulation_Set.set_domain(dim_range)
Emulation_Set.set_values(np.array( np.transpose([ np.random.beta(a=alpha, b=beta,
            size=num_samples_emulate_data_space) for i in range(dim_input) ]) )) # TODO multiple alpha, beta
Emulation_Discretization = sampler.compute_QoI_and_create_discretization(Emulation_Set)
samp.save_discretization(Emulation_Discretization, emul_filename)

# Create Data Space Discretizations
# ensure_path_exists(ref_sol_dir)
for M in M_values:
    Mval = M**(1 + (dim_input-1)*(data_discretization_type == 'reg') )    
    
    Partition_Set = samp.sample_set(dim_input)
    Partition_Set.set_domain(dim_range)
    if data_discretization_type == 'reg': # regular mesh for partition
        
        # NOTE: leads to errors with a NaN value in array
        # Partition_Set = samp.cartesian_sample_set(dim_input)
        # Partition_Set.set_domain(dim_range)
        # lin_mesh = []
        # for each_dim_range in dim_range:
        #     lin_mesh.append( np.linspace(each_dim_range[0], each_dim_range[1], M+1) )
        # Partition_Set.setup(lin_mesh)
        
        Partition_Set = bsam.regular_sample_set(Partition_Set, num_samples_per_dim = np.repeat(M, dim_input, axis=0))
    else: # random mesh for partition
        Partition_Set = bsam.random_sample_set('random',
                Partition_Set, num_samples = M)
    Partition_Discretization = sampler.compute_QoI_and_create_discretization(Partition_Set)
    
    part_filename = ref_sol_dir + 'Partition_Disc' '-' + \
            '%s_M_%d'%(data_discretization_type, Mval)
    
    samp.save_discretization(Partition_Discretization, part_filename)
print 'Data Space Discretizations Computed'


# Compute Reference Solutions
for BigN in BigN_values:
    BigNval = BigN**(1 + (dim_input-1)*(reference_mesh_type == 'reg') )
    ref_sol_dir_2 = ref_sol_dir + '%s_BigN_%d'%(reference_mesh_type, BigNval) + '/'
    ref_filename = 'Reference_Disc-' + '%s_BigN_%d'%(reference_mesh_type, BigNval) 
    Ref_Discretization = samp.load_discretization(ref_sol_dir_2 + ref_filename)
    # Emulation_Discretization = samp.load_discretization(emul_filename)
    for M in M_values:
        Mval = M**(1 + (dim_input-1)*(data_discretization_type == 'reg') )
        
        ref_sol_dir_3 = ref_sol_dir_2 + '%s_M_%d'%(data_discretization_type, Mval) + '/'
        ensure_path_exists(ref_sol_dir_3)
        
        part_filename = ref_sol_dir + 'Partition_Disc' '-' + \
                '%s_M_%d'%(data_discretization_type, Mval)
        Partition_Discretization = samp.load_discretization(part_filename)
        
        for sol_num in range(len(QoI_choice_list)):
            QoI_indices = QoI_choice_list[sol_num]
        
            filename = ref_sol_dir_3 + 'SolQoI_choice_%d'%(sol_num+1) + '-' + \
                    '%s_M_%d'%(data_discretization_type, Mval) + '_'  + \
                    '%s_BigN_%d'%(reference_mesh_type, BigNval)
            ref_solutions_filenames.append( filename )
            
            ref_discretization = invert_using(Ref_Discretization, 
                    Partition_Discretization, Emulation_Discretization, 
                    QoI_indices, Emulate = False)
                    
            samp.save_discretization(ref_discretization, filename)
# print ref_solutions_filenames
print 'Reference Solutions Computed'


# Compute discretizations for estimates to solution - multiple trials 
for N in N_values:
    Nval = N**(1 + (dim_input-1)*(estimate_mesh_type == 'reg') )
    temp_folder_name = '%s_N_%d'%(estimate_mesh_type, Nval)
    est_disc_dir = cwd + '/' + results_dir + '/' + sub_dirs[2] + '/' + temp_folder_name + '/' # all trial discretizations inside this folder
    ensure_path_exists(est_disc_dir)
    
    # np.random.seed(N)
    for trial in range(num_trials):
        filename = est_disc_dir + '%s_N_%d'%(estimate_mesh_type, Nval) + '_trial_%d'%(trial)
        est_discretizations_filenames.append(filename)
        Input_Samples = samp.sample_set(dim_input)
        Input_Samples.set_domain(dim_range)
        if estimate_mesh_type == 'reg': # regular mesh for solving problem
            # NOTE: leads to errors with a NaN value in array
            # Input_Samples = samp.cartesian_sample_set(dim_input)
            # Input_Samples.set_domain(dim_range)
            # lin_mesh = []
            # for each_dim_range in dim_range:
            #     lin_mesh.append( np.linspace(each_dim_range[0], each_dim_range[1], N+1) )
            # Input_Samples.setup(lin_mesh)
            
            Input_Samples = bsam.regular_sample_set(Input_Samples, num_samples_per_dim = np.repeat(N, dim_input, axis=0))
        else: # random mesh for solving problem (coarse meshes relative to the reference)
            Input_Samples = bsam.random_sample_set('random', Input_Samples, num_samples = N)
        Input_Samples.estimate_volume_mc()
        My_Discretization = sampler.compute_QoI_and_create_discretization(Input_Samples)
        if use_volumes:
            emulated_input_samples = samp.sample_set(dim_input)
            emulated_input_samples.set_domain(dim_range)
            emulated_input_samples = bsam.random_sample_set('random', emulated_input_samples, num_samples = num_emulated_input_samples)
            My_Discretization.set_emulated_input_sample_set(emulated_input_samples)
        samp.save_discretization(My_Discretization, filename)
        
        # Generate io_pointer for integration sets
        for int_mesh_num in range(len(I_values)):
            Ival = I_values[int_mesh_num]**(1 + (dim_input-1)*(integration_mesh_type == 'reg') )
            filename = est_disc_dir + 'ptr_from_' + '%s_I_%d'%(integration_mesh_type, Ival) + \
                    '_to_' + '%s_N_%d'%(estimate_mesh_type, Nval) + '_trial_%d'%(trial)
            int_mesh_set = samp.load_sample_set(integration_sets_filenames[int_mesh_num])  
            (_, ptr) = Input_Samples.query(int_mesh_set._values)
            np.save(filename, ptr)
            est_sol_int_ptrs_filenames.append( filename + '.npy')
# print est_discretizations_filenames
print 'Estimated Solution Discretizations Computed, Pointers from Integration Sets Created'


# estimated solutions
for sol_num in range(len(QoI_choice_list)): # for each choice of skewed map
    QoI_indices = QoI_choice_list[sol_num]
    est_sol_dir_2 = est_sol_dir + 'QoI_choice_%d'%(sol_num+1) + '/'
    
    print 'Solving Inverse Problems for Skew = %d'%(skew_range[sol_num])
    for M in M_values: # for each data-space discretization
        Mval = M**(1 + (dim_input-1)*(data_discretization_type == 'reg') )
        part_filename = ref_sol_dir + 'Partition_Disc' '-' + \
                '%s_M_%d'%(data_discretization_type, Mval)
        Partition_Discretization = samp.load_discretization(part_filename)
        est_sol_dir_3 = est_sol_dir_2 + '%s_M_%d'%(data_discretization_type, Mval) + '/'
        
        print '\t With M = %d'%(Mval)
        for N in N_values:
            Nval = N**(1 + (estimate_mesh_type == 'reg') )
            est_sol_dir_4 = est_sol_dir_3 + '%s_N_%d'%(estimate_mesh_type, Nval) + '/' # all trial sols inside this folder
            ensure_path_exists(est_sol_dir_4)
            
            for trial in range(num_trials):
                filename = est_sol_dir_4 + 'SolQoI_choice_%d'%(sol_num+1) + '-' + \
                '%s_M_%d'%(data_discretization_type, Mval) + \
                '_' + '%s_N_%d'%(estimate_mesh_type, Nval) + '_trial_%d'%(trial)
                est_solutions_filenames.append(filename)
                
                load_est_disc_dir = cwd + '/' + results_dir + '/' + sub_dirs[2] + '/' + \
                        '%s_N_%d'%(estimate_mesh_type, Nval) + '/' + \
                        '%s_N_%d'%(estimate_mesh_type, Nval) + '_trial_%d'%(trial)
                My_Discretization = samp.load_discretization(load_est_disc_dir)
                
                my_discretization = invert_using(My_Discretization, 
                        Partition_Discretization, Emulation_Discretization,
                        QoI_indices, Emulate = use_volumes)
                
                samp.save_discretization(my_discretization, filename)
            print '\t \t %d Trials Completed for N = %d'%(num_trials, Nval)
            
# print est_solutions_filenames
paths_of_interest = np.array([integration_sets_filenames, ref_solutions_filenames, est_solutions_filenames, ref_sol_int_ptrs_filenames, est_sol_int_ptrs_filenames])
np.save(cwd + '/' + results_dir + '/' + 'paths', paths_of_interest)