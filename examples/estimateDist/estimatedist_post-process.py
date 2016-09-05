from estimatedist_setup import *


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
                        
            
            
            for N in N_values:
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
                print 'Computed for BigN = %8d, I = %6d, M = %3d, N = %6d'%(BigNval, Ival, Mval, Nval)
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
