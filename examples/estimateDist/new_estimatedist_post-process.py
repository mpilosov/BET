from new_estimatedist_setup import *


for BigN in BigN_values: # reference solution resolution
    BigNval = BigN**(1 + (dim_input-1)*(reference_mesh_type == 'reg') )
    ref_sol_dir_2 = ref_sol_dir + '%s_BigN_%d'%(reference_mesh_type, BigNval) + '/'
    data_dir_2 = data_dir+ '%s_BigN_%d'%(reference_mesh_type, BigNval)
    
    for M in M_values: # data space discretization mesh
        Mval = M**(1 + (dim_input-1)*(data_discretization_type == 'reg') )
        ref_sol_dir_3 = ref_sol_dir_2 + '%s_M_%d'%(data_discretization_type, Mval) + '/'
        data_dir_3 = data_dir_2 + '%s_M_%d'%(data_discretization_type, Mval) + '/'
        ensure_path_exists(data_dir_3)
        for I in I_values: # integration mesh
            Ival = I**(1 + (dim_input-1)*(integration_mesh_type == 'reg') )
            
            int_set_filename = cwd +  '/' + results_dir + '/' + sub_dirs[1] + \
                    '/' + '%s_IntSet_%d'%(integration_mesh_type, Ival)
            Integration_Set = samp.load_sample_set(int_set_filename)
            
            data_dict = { (N**(1 + (dim_input-1)*(estimate_mesh_type == 'reg') )) : {} for N in N_values }
            
            for sol_num in range(len(QoI_choice_list)): # number of possible QoIs
                QoI_indices = QoI_choice_list[sol_num]
                
                ref_sol_filename = ref_sol_dir_3 + 'SolQoI_choice_%d'%(sol_num+1) + '-' + \
                        '%s_M_%d'%(data_discretization_type, Mval) + '_'  + \
                        '%s_BigN_%d'%(reference_mesh_type, BigNval)
                ref_ptr_filename = ref_sol_dir_2 + 'ptr_from_' + \
                        '%s_I_%d'%(integration_mesh_type, Ival) + '_to_' + \
                         '%s_BigN_%d'%(reference_mesh_type, BigNval) + '.npy'
                               
                Ref_Disc = samp.load_discretization(ref_sol_filename)
                ref_ptr = np.load(ref_ptr_filename)
                        
                
                est_sol_dir_2 = est_sol_dir + 'QoI_choice_%d'%(sol_num+1) + '/'        
                est_sol_dir_3 = est_sol_dir_2 + '%s_M_%d'%(data_discretization_type, Mval) + '/'
                
                for N in N_values:
                    Nval = N**(1 + (dim_input-1)*(estimate_mesh_type == 'reg') )
                    est_sol_dir_4 = est_sol_dir_3 + '%s_N_%d'%(estimate_mesh_type, Nval) + '/' # all trial sols inside this folder
                    
                    temp_array = np.zeros( (num_trials, len(QoI_choice_list)) )
                    for trial in range(num_trials):
                        est_sol_filename = est_sol_dir_4 + 'SolQoI_choice_%d'%(sol_num+1) + '-' + \
                        '%s_M_%d'%(data_discretization_type, Mval) + \
                        '_' + '%s_N_%d'%(estimate_mesh_type, Nval) + '_trial_%d'%(trial)
                        
                        est_ptr_filename = cwd + '/' + results_dir + '/' + sub_dirs[2] + '/' + \
                                '%s_N_%d'%(estimate_mesh_type, Nval) + '/' + \
                                'ptr_from_' + '%s_I_%d'%(integration_mesh_type, Ival) + \
                                '_to_' + '%s_N_%d'%(estimate_mesh_type, Nval) + '_trial_%d'%(trial) + '.npy'
                                        
                        Est_Disc = samp.load_discretization(est_sol_filename)
                        est_ptr = np.load(est_ptr_filename)
                        
                        temp_array[trial, sol_num] = mc_Hellinger(Integration_Set,
                                Ref_Disc, ref_ptr, 
                                Est_Disc, est_ptr)
                    data_dict[Nval]['data'] = temp_array
                    data_dict[Nval]['stats'] = [ np.mean(temp_array, axis=0), np.var(temp_array, axis=0) ]           
                    
            # save data object
            data_filename = data_dir_3 + 'Data-' + '%s_BigN_%d'%(reference_mesh_type, BigNval) + \
                    '_' + '%s_M_%d'%(data_discretization_type, Mval) + '_' + \
                    '%s_I_%d'%(integration_mesh_type, Ival)
            np.save(data_filename, data_dict)
