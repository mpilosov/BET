import sys
# ref_input_num = int(sys.argv[1])
import pyprind
from estimatedist_funs import *
from matplotlib import pyplot as plt
# from heatrod_model import make_model

from estimatedist_setup import *
'''

dim_input = 2
dim_output = 2

dim_range = np.array([[0.01, 0.2],[0.01, 0.2]]) # parameter space domain 
temp_locs_list = [0.125, 0.9, 0.7] # list of QoI locations 
QoI_choice_list = [ [0, n] for n in range(1,len(temp_locs_list))] # first location fixed.

my_model = make_model(temp_locs_list)
sampler = bsam.sampler(my_model)

alpha = 1 # TODO add functionality for multiple alpha, betas - one per input_dim
beta = 1

# regular or random for all of them.
data_discretization_type = 'reg'
M_values = [1]
num_samples_emulate_data_space = 1E5 # TODO append to Emulation_Discretization filename

reference_mesh_type = 'reg'
BigN_values = [200]

estimate_mesh_type = 'rand'
N_values = [20*2**n for n in range(8)]
use_volumes = False # use calculateP.prob_with_emulated_volumes or just calculateP.prob - this uses emulated points
num_emulated_input_samples = 1E6

integration_mesh_type =  'rand'
I_values = [1E5]
num_trials = 50

kappa_lin = np.linspace(0.01, 0.2, 5)[1:-1] # reference kappas
kappa_ref_locs = np.array([[k1,k2] for k1 in kappa_lin for k2 in kappa_lin]) # like meshgrid, only easier to parse

ref_input_num = 4 # 4 is middle
ref_input = np.array([kappa_ref_locs[ref_input_num]]) # SUPER CONFUSING INDEXING NONSENSE - line 45 in estimatedist_funs
Qref =  my_model(ref_input)[0] # BEST I COULD DO TO GET AROUND IT
rect = 0.1 # currently scale, not size

recover = False

show_title = False
label_fsize = 20
tick_fsize = 14
legend_fsize = 14

## Initial Run
# create_int_sets = True
# create_data_discs = False
# compute_emulated_set = False
# create_ref_disc = True
# create_est_discs = True
# compute_ref_sol = True
# compute_est_sol = True

## Post-Processing - change description of uncertainty on output
create_int_sets = False
create_data_discs = False
compute_emulated_set = False
create_ref_disc = False
create_est_discs = False
compute_ref_sol = True
compute_est_sol = True

cwd = os.getcwd()
results_dir = 'results_heatrod_2'
sub_dirs = ['postprocess_rectscale_%d'%(1+ref_input_num), 'integration_sets', 'est_discretizations', 'est_solutions', 'ref_solutions']
ref_sol_dir =  cwd + '/' + results_dir + '/' + sub_dirs[4] + '/' # commonly used
est_sol_dir =  cwd + '/' + results_dir + '/' + sub_dirs[3] + '/' # commonly used
data_dir = cwd + '/' + results_dir + '/' + sub_dirs[0] + '/'

####################################################################################
'''
integration_sets_filenames = []
ref_solutions_filenames = []
est_solutions_filenames = []
# TODO: can probably delete these lists here. Make sure they're unused.

est_discretizations_filenames = []  # dont think I need this one

emul_filename = ref_sol_dir + \
    'Emulation_Discretization_%d' % (num_samples_emulate_data_space)

# Create Integration Sets
# NOTE Dependencies: integration_mesh_type, I_values, dim_input, dim_range, temp_dir, results_dir, sub_dirs
if create_int_sets:
    for I in I_values:  # resolution of integration mesh
        Ival = I**(1 + (dim_input-1)*(integration_mesh_type == 'reg'))
        temp_dir = cwd + '/' + results_dir + '/' + sub_dirs[1] + '/'
        ensure_path_exists(temp_dir)
        filename = temp_dir + '%s_IntSet_%d' % (integration_mesh_type, Ival)
        # integration_sets_filenames.append(filename)

        integration_sample_set = samp.sample_set(dim_input)
        integration_sample_set.set_domain(dim_range)
        if integration_mesh_type == 'reg':  # regular mesh for monte-carlo integration

            # NOTE: not sure if this provides any speedup because this is the input set
            # (into the computationally expensive KDTree being queried) is the reference mesh.
            # speedup happens with the reference mesh being a cartesian_sample_set.

            # integration_sample_set = samp.cartesian_sample_set(dim_input)
            # integration_sample_set.set_domain(dim_range)
            # lin_mesh = []
            # for each_dim_range in dim_range:
            #     lin_mesh.append( np.linspace(each_dim_range[0], each_dim_range[1], I+1) )
            # integration_sample_set.setup(lin_mesh)

            integration_sample_set = bsam.regular_sample_set(
                integration_sample_set, num_samples_per_dim=np.repeat(I, dim_input, axis=0))
        else:  # random mesh for monte-carlo integration
            integration_sample_set = bsam.random_sample_set('random',
                                                            integration_sample_set, num_samples=I)
        integration_sample_set.estimate_volume_mc()
        samp.save_sample_set(integration_sample_set, filename)
    print('Integration Sets Computed')


# Create Reference Discretizations
# NOTE Dependencies: reference_mesh_type, BigN_values, dim_input, dim_range, ref_sol_dir, sampler
if create_ref_disc:
    for BigN in BigN_values:
        BigNval = BigN**(1 + (dim_input-1)*(reference_mesh_type == 'reg'))
        ref_sol_dir_2 = ref_sol_dir + \
            '%s_BigN_%d' % (reference_mesh_type, BigNval) + '/'
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

            ref_sample_set = bsam.regular_sample_set(
                ref_sample_set, num_samples_per_dim=np.repeat(BigN, dim_input, axis=0))
        else:
            ref_sample_set = bsam.random_sample_set('random',
                                                    ref_sample_set, num_samples=BigN)
        ref_sample_set.estimate_volume_mc()

        Ref_Discretization = sampler.compute_QoI_and_create_discretization(
            ref_sample_set)

        ref_filename = ref_sol_dir_2 + 'Reference_Disc-' + \
            '%s_BigN_%d' % (reference_mesh_type, BigNval)
        samp.save_discretization(Ref_Discretization, ref_filename)
    print('Reference Discretizations Computed\n')
    print('\t You can now run python estimatedist_build_pointers.py')


# Create one emulation set instead of one for each M.
# NOTE Dependencies: dim_input, dim_range, alpha, beta, emul_filename, sampler
if compute_emulated_set:
    # generate emulated set from true distribution
    Emulation_Set = samp.sample_set(dim_input)
    Emulation_Set.set_domain(dim_range)
    Emulation_Set.set_values(np.array(np.transpose([np.random.beta(a=alpha, b=beta,
                                                                   size=num_samples_emulate_data_space) for i in range(dim_input)])))  # TODO multiple alpha, beta
    Emulation_Discretization = sampler.compute_QoI_and_create_discretization(
        Emulation_Set)
    samp.save_discretization(Emulation_Discretization, emul_filename)


# Create Data Space Discretizations
# NOTE Dependencies: data_discretization_type, M_values, dim_input, dim_range, sampler
if create_data_discs:
    for M in M_values:
        Mval = M**(1 + (dim_input-1)*(data_discretization_type == 'reg'))

        Partition_Set = samp.sample_set(dim_input)
        Partition_Set.set_domain(dim_range)
        if data_discretization_type == 'reg':  # regular mesh for partition

            # NOTE: leads to errors with a NaN value in array
            # Partition_Set = samp.cartesian_sample_set(dim_input)
            # Partition_Set.set_domain(dim_range)
            # lin_mesh = []
            # for each_dim_range in dim_range:
            #     lin_mesh.append( np.linspace(each_dim_range[0], each_dim_range[1], M+1) )
            # Partition_Set.setup(lin_mesh)

            Partition_Set = bsam.regular_sample_set(
                Partition_Set, num_samples_per_dim=np.repeat(M, dim_input, axis=0))
        else:  # random mesh for partition
            Partition_Set = bsam.random_sample_set('random',
                                                   Partition_Set, num_samples=M)
        Partition_Discretization = sampler.compute_QoI_and_create_discretization(
            Partition_Set)

        part_filename = ref_sol_dir + 'Partition_Disc' '-' + \
            '%s_M_%d' % (data_discretization_type, Mval)

        samp.save_discretization(Partition_Discretization, part_filename)
    print('Data Space Discretizations Created')


# Create discretizations for estimates to solution - multiple trials
# NOTE: Dependencies: N_values, estimate_mesh_type, dim_input, dim_range, results_dir, sub_dirs, num_trials, use_volumes, num_emulated_input_samples
if create_est_discs:
    for N in N_values:
        Nval = N**(1 + (dim_input-1)*(estimate_mesh_type == 'reg'))
        temp_folder_name = '%s_N_%d' % (estimate_mesh_type, Nval)
        # all trial discretizations inside this folder
        est_disc_dir = cwd + '/' + results_dir + '/' + \
            sub_dirs[2] + '/' + temp_folder_name + '/'
        ensure_path_exists(est_disc_dir)

        # np.random.seed(N)
        for trial in range(num_trials):
            filename = est_disc_dir + \
                '%s_N_%d' % (estimate_mesh_type, Nval) + '_trial_%d' % (trial)
            # est_discretizations_filenames.append(filename)
            Input_Samples = samp.sample_set(dim_input)
            Input_Samples.set_domain(dim_range)
            if estimate_mesh_type == 'reg':  # regular mesh for solving problem
                # NOTE: leads to errors with a NaN value in array
                # Input_Samples = samp.cartesian_sample_set(dim_input)
                # Input_Samples.set_domain(dim_range)
                # lin_mesh = []
                # for each_dim_range in dim_range:
                #     lin_mesh.append( np.linspace(each_dim_range[0], each_dim_range[1], N+1) )
                # Input_Samples.setup(lin_mesh)

                Input_Samples = bsam.regular_sample_set(
                    Input_Samples, num_samples_per_dim=np.repeat(N, dim_input, axis=0))
            # random mesh for solving problem (coarse meshes relative to the reference)
            else:
                Input_Samples = bsam.random_sample_set(
                    'random', Input_Samples, num_samples=N)
            Input_Samples.estimate_volume_mc()

            My_Discretization = sampler.compute_QoI_and_create_discretization(
                Input_Samples)
            if use_volumes:
                emulated_input_samples = samp.sample_set(dim_input)
                emulated_input_samples.set_domain(dim_range)
                emulated_input_samples = bsam.random_sample_set(
                    'random', emulated_input_samples, num_samples=num_emulated_input_samples)
                My_Discretization.set_emulated_input_sample_set(
                    emulated_input_samples)
            samp.save_discretization(My_Discretization, filename)
        print('\t Estimated Solution Discretizations for N = %d Computed' % (Nval))
    # print est_discretizations_filenames


# Compute Reference Solutions
# NOTE: Dependencies: reference_mesh_type, BigN_values, dim_input, ref_sol_dir, M_values, QoI_choice_list (output_dim can replace this), emul_filename
if compute_ref_sol:
    for BigN in BigN_values:
        BigNval = BigN**(1 + (dim_input-1)*(reference_mesh_type == 'reg'))
        ref_sol_dir_2 = ref_sol_dir + \
            '%s_BigN_%d' % (reference_mesh_type, BigNval) + '/'
        ref_filename = ref_sol_dir_2 + 'Reference_Disc-' + \
            '%s_BigN_%d' % (reference_mesh_type, BigNval)
        Ref_Discretization = samp.load_discretization(ref_filename)
        if recover:
            Emulation_Discretization = samp.load_discretization(emul_filename)
        for M in M_values:
            Mval = M**(1 + (dim_input-1)*(data_discretization_type == 'reg'))

            ref_sol_dir_3 = ref_sol_dir_2 + \
                '%s_M_%d' % (data_discretization_type, Mval) + '/'
            ensure_path_exists(ref_sol_dir_3)

            if recover:
                part_filename = ref_sol_dir + 'Partition_Disc' '-' + \
                    '%s_M_%d' % (data_discretization_type, Mval)
                Partition_Discretization = samp.load_discretization(
                    part_filename)

            for sol_num in range(len(QoI_choice_list)):
                QoI_indices = QoI_choice_list[sol_num]

                if recover:
                    ref_discretization = invert_using(Ref_Discretization,
                                                      Partition_Discretization, Emulation_Discretization,
                                                      QoI_indices, Emulate=False)
                    filename = ref_sol_dir_3 + 'SolQoI_choice_%d' % (sol_num+1) + '-' + \
                        '%s_RM_%d' % (data_discretization_type, Mval) + '_' + \
                        '%s_BigN_%d' % (reference_mesh_type, BigNval)
                else:
                    ref_discretization = invert_rect_using(Ref_Discretization,
                                                           QoI_indices, Qref, rect,
                                                           cells_per_dimension=M, Emulate=use_volumes)
                    filename = ref_sol_dir_3 + 'SolQoI_choice_%d' % (sol_num+1) + '-' + \
                        '%s_M_%d' % (data_discretization_type, Mval) + '_' + \
                        '%s_BigN_%d' % (reference_mesh_type, BigNval)

                samp.save_discretization(ref_discretization, filename)
                # ref_solutions_filenames.append( filename )
    # print ref_solutions_filenames
    print('Reference Solutions Computed')


# Compute Estimated Solutions
# NOTE: Dependencies: M_values, N_values, est_sol_dir, QoI_choice_list, use_volumes, results_dir, sub_dirs
if compute_est_sol:
    if recover:
        Emulation_Discretization = samp.load_discretization(emul_filename)
    for sol_num in range(len(QoI_choice_list)):  # for each choice of skewed map
        QoI_indices = QoI_choice_list[sol_num]
        est_sol_dir_2 = est_sol_dir + 'QoI_choice_%d' % (sol_num+1) + '/'

        print('Solving Inverse Problems for QoI_%d' % (sol_num))
        for M in M_values:  # for each data-space discretization
            Mval = M**(1 + (dim_input-1)*(data_discretization_type == 'reg'))
            if recover:
                part_filename = ref_sol_dir + 'Partition_Disc' '-' + \
                    '%s_M_%d' % (data_discretization_type, Mval)
                Partition_Discretization = samp.load_discretization(
                    part_filename)

            if recover:
                est_sol_dir_3 = est_sol_dir_2 + \
                    '%s_RM_%d' % (data_discretization_type, Mval) + '/'
            else:
                est_sol_dir_3 = est_sol_dir_2 + \
                    '%s_M_%d' % (data_discretization_type, Mval) + '/'

            print('\t With M = %d' % (Mval))
            for N in pyprind.prog_bar(N_values):
                Nval = N**(1 + (estimate_mesh_type == 'reg'))
                # all trial sols inside this folder
                est_sol_dir_4 = est_sol_dir_3 + \
                    '%s_N_%d' % (estimate_mesh_type, Nval) + '/'
                ensure_path_exists(est_sol_dir_4)

                for trial in range(num_trials):
                    load_est_disc_dir = cwd + '/' + results_dir + '/' + sub_dirs[2] + '/' + \
                        '%s_N_%d' % (estimate_mesh_type, Nval) + '/' + \
                        '%s_N_%d' % (estimate_mesh_type, Nval) + \
                        '_trial_%d' % (trial)
                    My_Discretization = samp.load_discretization(
                        load_est_disc_dir)
                    My_Discretization._input_sample_set.check_num()

                    if recover:
                        my_discretization = invert_using(My_Discretization,
                                                         Partition_Discretization, Emulation_Discretization,
                                                         QoI_indices, Emulate=use_volumes)
                        filename = est_sol_dir_4 + 'SolQoI_choice_%d' % (sol_num+1) + '-' + \
                            '%s_RM_%d' % (data_discretization_type, Mval) + \
                            '_' + '%s_N_%d' % (estimate_mesh_type,
                                               Nval) + '_trial_%d' % (trial)
                    else:
                        my_discretization = invert_rect_using(My_Discretization,
                                                              QoI_indices, Qref, rect,
                                                              cells_per_dimension=M, Emulate=use_volumes)
                        filename = est_sol_dir_4 + 'SolQoI_choice_%d' % (sol_num+1) + '-' + \
                            '%s_M_%d' % (data_discretization_type, Mval) + \
                            '_' + '%s_N_%d' % (estimate_mesh_type,
                                               Nval) + '_trial_%d' % (trial)
                    samp.save_discretization(my_discretization, filename)
                    # est_solutions_filenames.append(filename)

                # print '\t \t %d Trials Completed for N = %d'%(num_trials, Nval)

# print est_solutions_filenames
# paths_of_interest = np.array([integration_sets_filenames, ref_solutions_filenames, est_solutions_filenames, ref_sol_int_ptrs_filenames, est_sol_int_ptrs_filenames])
# np.save(cwd + '/' + results_dir + '/' + 'paths', paths_of_interest)
'''

for BigN in BigN_values: # reference solution resolution
    BigNval = BigN**(1 + (dim_input-1)*(reference_mesh_type == 'reg') )
    ref_sol_dir_2 = ref_sol_dir + '%s_BigN_%d'%(reference_mesh_type, BigNval) + '/'
    data_dir_2 = data_dir + '%s_BigN_%d'%(reference_mesh_type, BigNval) + '/'
    
    for M in M_values: # data space discretization mesh
        Mval = M**(1 + (dim_input-1)*(data_discretization_type == 'reg') )
        
        if recover:
            ref_sol_dir_3 = ref_sol_dir_2 + '%s_RM_%d'%(data_discretization_type, Mval) + '/'
            data_dir_3 = data_dir_2 + '%s_RM_%d'%(data_discretization_type, Mval) + '/'
        else:
            ref_sol_dir_3 = ref_sol_dir_2 + '%s_M_%d'%(data_discretization_type, Mval) + '/'
            data_dir_3 = data_dir_2 + '%s_M_%d'%(data_discretization_type, Mval) + '/'
        
        ensure_path_exists(data_dir_3)
        for I in I_values: # integration mesh
            Ival = I**(1 + (dim_input-1)*(integration_mesh_type == 'reg') )
            
            int_set_filename = cwd +  '/' + results_dir + '/' + sub_dirs[1] + \
                    '/' + '%s_IntSet_%d'%(integration_mesh_type, Ival)
            Integration_Set = samp.load_sample_set(int_set_filename)
            
            data_dict = { (N**(1 + (dim_input-1)*(estimate_mesh_type == 'reg') )) : {} for N in N_values }
            
            ref_ptr_filename = ref_sol_dir_2 + 'ptr_from_' + \
                    '%s_I_%d'%(integration_mesh_type, Ival) + '_to_' + \
                     '%s_BigN_%d'%(reference_mesh_type, BigNval) + '.npy'
            ref_ptr = np.load(ref_ptr_filename)   
                        
            
            
            for N in pyprind.prog_percent(N_values):
                Nval = N**(1 + (dim_input-1)*(estimate_mesh_type == 'reg') )
                
                temp_array = np.zeros( (num_trials, len(QoI_choice_list)) )
                
                for trial in range(num_trials):
                    
                    
                    est_ptr_filename = cwd + '/' + results_dir + '/' + sub_dirs[2] + '/' + \
                            '%s_N_%d'%(estimate_mesh_type, Nval) + '/' + \
                            'ptr_from_' + '%s_I_%d'%(integration_mesh_type, Ival) + \
                            '_to_' + '%s_N_%d'%(estimate_mesh_type, Nval) + '_trial_%d'%(trial) + '.npy'
                                    
                    
                    est_ptr = np.load(est_ptr_filename)
                    
                    for sol_num in range(len(QoI_choice_list)): # number of possible QoIs
                        est_sol_dir_2 = est_sol_dir + 'QoI_choice_%d'%(sol_num+1) + '/'        
                        est_sol_dir_3 = est_sol_dir_2 + '%s_M_%d'%(data_discretization_type, Mval) + '/'
                        est_sol_dir_4 = est_sol_dir_3 + '%s_N_%d'%(estimate_mesh_type, Nval) + '/' # all trial sols inside this folder
                        
                        if recover:
                            est_sol_filename = est_sol_dir_4 + 'SolQoI_choice_%d'%(sol_num+1) + '-' + \
                            '%s_RM_%d'%(data_discretization_type, Mval) + \
                            '_' + '%s_N_%d'%(estimate_mesh_type, Nval) + '_trial_%d'%(trial)
                            
                            ref_sol_filename = ref_sol_dir_3 + 'SolQoI_choice_%d'%(sol_num+1) + '-' + \
                                    '%s_RM_%d'%(data_discretization_type, Mval) + '_'  + \
                                    '%s_BigN_%d'%(reference_mesh_type, BigNval)
                        else:
                            est_sol_filename = est_sol_dir_4 + 'SolQoI_choice_%d'%(sol_num+1) + '-' + \
                            '%s_M_%d'%(data_discretization_type, Mval) + \
                            '_' + '%s_N_%d'%(estimate_mesh_type, Nval) + '_trial_%d'%(trial)
                            
                            ref_sol_filename = ref_sol_dir_3 + 'SolQoI_choice_%d'%(sol_num+1) + '-' + \
                                    '%s_M_%d'%(data_discretization_type, Mval) + '_'  + \
                                    '%s_BigN_%d'%(reference_mesh_type, BigNval)
                                 
                        Ref_Disc = samp.load_discretization(ref_sol_filename)
                        Est_Disc = samp.load_discretization(est_sol_filename)
                        # Ref_Disc2 = Ref_Disc.copy()
                        # zero_probs = np.array(Ref_Disc._input_sample_set._probabilities < .0004)
                        # # print len(zero_probs[zero_probs == True])
                        # # print np.sort(Ref_Disc._input_sample_set._probabilities)
                        # Ref_Disc2._input_sample_set._probabilities[zero_probs] = 1./len(zero_probs[zero_probs == True])
                        # Ref_Disc2._input_sample_set._probabilities[~zero_probs] = 0
                        # Ref_Disc._input_sample_set._probabilities[~zero_probs] = 1./len(zero_probs[zero_probs == False])
                        # Ref_Disc._input_sample_set._probabilities[zero_probs] = 0
                        temp_array[trial, sol_num] = mc_Hellinger(Integration_Set,
                                Ref_Disc, ref_ptr, 
                                Est_Disc, est_ptr)
                                # Ref_Disc2, ref_ptr)
                # print 'Computed for BigN = %8d, I = %6d, M = %3d, N = %6d'%(BigNval, Ival, Mval, Nval)
                # print temp_array
                data_dict[Nval]['data'] = temp_array
                data_dict[Nval]['stats'] = [ np.mean(temp_array, axis=0), np.var(temp_array, axis=0) ]           
                
            # save data object
            if recover:
                data_filename = data_dir_3 + 'Data-' + '%s_BigN_%d'%(reference_mesh_type, BigNval) + \
                        '_' + '%s_RM_%d'%(data_discretization_type, Mval) + '_' + \
                        '%s_I_%d'%(integration_mesh_type, Ival)
            else:
                data_filename = data_dir_3 + 'Data-' + '%s_BigN_%d'%(reference_mesh_type, BigNval) + \
                        '_' + '%s_M_%d'%(data_discretization_type, Mval) + '_' + \
                        '%s_I_%d'%(integration_mesh_type, Ival)
            np.save(data_filename, data_dict)


mean_or_var = 'Mean'

line_colors = np.linspace(0.8, 0, len(temp_locs_list)) # LIGHT TO DARK - LOW to HIGH Theta

for BigN in BigN_values: # reference solution resolution
    BigNval = BigN**(1 + (dim_input-1)*(reference_mesh_type == 'reg') )
    data_dir_2 = data_dir + '%s_BigN_%d'%(reference_mesh_type, BigNval) + '/'
    for M in M_values: # data space discretization mesh
        Mval = M**(1 + (dim_input-1)*(data_discretization_type == 'reg') )
        if recover:
            data_dir_3 = data_dir_2 + '%s_RM_%d'%(data_discretization_type, Mval) + '/'
        else:            
            data_dir_3 = data_dir_2 + '%s_M_%d'%(data_discretization_type, Mval) + '/'
        
        if not recover:
            print 'rect = %f'%rect, 'lambda_ref = ', ref_input[:], '\n'
        for I in I_values: # integration mesh
            Ival = I**(1 + (dim_input-1)*(integration_mesh_type == 'reg') )
            if recover:
                data_filename = data_dir_3 + 'Data-' + '%s_BigN_%d'%(reference_mesh_type, BigNval) + \
                        '_' + '%s_RM_%d'%(data_discretization_type, Mval) + '_' + \
                        '%s_I_%d'%(integration_mesh_type, Ival)
            else:
                data_filename = data_dir_3 + 'Data-' + '%s_BigN_%d'%(reference_mesh_type, BigNval) + \
                        '_' + '%s_M_%d'%(data_discretization_type, Mval) + '_' + \
                        '%s_I_%d'%(integration_mesh_type, Ival)
                    
            D = np.load(data_filename + '.npy')
            D = D.item()
            # data_dict = { (N**(1 + (dim_input-1)*(estimate_mesh_type == 'reg') )) : {} for N in N_values }
            if mean_or_var == 'Mean':
                data_for_M = np.array([ D[ N**(1 + (dim_input-1)*(estimate_mesh_type == 'reg') ) ]['stats'][0] for N in N_values])
            else:
                data_for_M = np.array([ D[ N**(1 + (dim_input-1)*(estimate_mesh_type == 'reg') ) ]['stats'][1] for N in N_values])
            
            alpha = ['a', 'b','c','d','e','f','g']
            
            print 'Integration Mesh I = %d'%Ival, 'Data Discretization M = %d'%Mval
            
            str1 = '\\begin{table}[h!]\n\\begin{tabular}{ c '
            for qoi_idx in range(len(QoI_choice_list)):
                str1 += '| c ' 
            str1 += '}'
            
            str1 += '\nN'
            for qoi_idx in range(len(QoI_choice_list)):
                str1 += ' & $Q^{(%s)}$'%alpha[qoi_idx]
            str1 += '\\\\'  + ' \\hline \\hline\n'
            for i in range(len(N_values)):
                N = N_values[i]
                Nval = N**(1 + (dim_input-1)*(estimate_mesh_type == 'reg') )
                str1 += '$%d$'%Nval
                for j in range(len(QoI_choice_list)):
                    str1 += ' & $%2.2E$'%data_for_M[i][j]
                str1 += '\\\\ \\hline \n \n'
            str1 += '\\end{tabular}\n\\end{table}'
            print str1
            
            print '\n\n'
            if recover:
                new_data_filename = data_dir_3 + 'Plot-' + '%s_BigN_%d'%(reference_mesh_type, BigNval) + \
                        '_' + '%s_RM_%d'%(data_discretization_type, Mval) + '_' + \
                        '%s_I_%d'%(integration_mesh_type, Ival) + '.eps'
            else:
                new_data_filename = data_dir_3 + 'Plot-' + '%s_BigN_%d'%(reference_mesh_type, BigNval) + \
                        '_' + '%s_M_%d'%(data_discretization_type, Mval) + '_' + \
                        '%s_I_%d'%(integration_mesh_type, Ival) + '.eps'
            plt.cla()
            lines = []
            
            ref_vec = np.array([np.min(N_values), np.max(N_values)])
            plt.plot(ref_vec,10./np.power(ref_vec, 1.0/2.0), linewidth=1.0, ls = '-', color = 'k')

            for qoi_idx in range(len(QoI_choice_list)):
                lines.append( plt.plot(N_values, data_for_M[:,qoi_idx], 'h', label = 'Q(%s)'%alpha[qoi_idx]) ) # NOTE not sure if label is working correctly
                plt.setp(lines[qoi_idx], linewidth=1.0,ls='--')
                if qoi_idx == 0:
                    plt.setp(lines, linewidth=1.0,ls='-')
                plt.setp(lines[qoi_idx], color = np.repeat(line_colors[qoi_idx],3,axis=0) )
            if show_title:
                if recover:
                    plt.title('Hellinger Distance with I = %d\n BigN = %d, M = %d'%(Ival, BigNval, Mval))
                else: 
                    plt.title('Hellinger Distances (I = %d, BigN = %d) for the\nParameter i.d. Problem w/ rect = %s, M = %d'%(Ival, BigNval, rect, Mval))
            
            plt.xlabel('Number of Samples', fontsize=label_fsize)
            # plt.ylabel('Hellinger Distance\n (%dE5 MC samples)'%(Ival/1E5), fontsize=label_fsize)
            plt.ylabel('Hellinger Distance', fontsize=label_fsize)
            plt.xscale('log')
            plt.yscale('log')
            plt.xticks(fontsize=tick_fsize)
            plt.yticks(fontsize=tick_fsize)
            plt.gcf().subplots_adjust(bottom=0.125,left=0.125)
            
            # legend_strings = [ '$Q^{(%s)}$'%alpha[qoi_idx] for qoi_idx in range(len(QoI_choice_list))]
            leg = ['MC Conv. Rate']
            for qoi_idx in range(len(QoI_choice_list)):
                leg.append( '$Q^{(%s)}$'%alpha[qoi_idx] )
                
            plt.legend(leg, loc = 'lower left', fontsize = legend_fsize) # NOTE: manually creating legend
            # plt.axis([20, 7500, 5E-3, 1])
            plt.savefig(new_data_filename)
'''
