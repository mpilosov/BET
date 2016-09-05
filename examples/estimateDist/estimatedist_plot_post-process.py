from estimatedist_setup import *
from matplotlib import pyplot as plt

mean_or_var = 'Mean'
# alpha = 1 # TODO add functionality for multiple alpha, betas - one per input_dim
# beta = 1
# 
# # regular or random for all of them.
# data_discretization_type = 'reg'
# M_values = [2, 4, 5, 10, 20]
# num_samples_emulate_data_space = 1E4
# 
# reference_mesh_type = 'reg'
# BigN_values = [200]
# # BigN_values = [1E5]
# 
# estimate_mesh_type = 'rand'
# N_values = [25*2**n for n in range(10)]  # NOTE when N > BigN, error shoots up.
# # N_values = [4, 16, 25, 100, 400, 2500, 10000 ]
# use_volumes = True # use calculateP.prob_with_emulated_volumes or just calculateP.prob - this uses emulated points
# num_emulated_input_samples = 1E5
# 
# integration_mesh_type =  'rand'
# I_values = [1E4] # map(int, [1E3, 1E4, 1E5]) 
# 
# num_trials = 10
# N_values = [2,5,20]

line_colors = np.linspace(0.8, 0, len(skew_range)) # LIGHT TO DARK - LOW to HIGH Theta

for BigN in BigN_values: # reference solution resolution
    BigNval = BigN**(1 + (dim_input-1)*(reference_mesh_type == 'reg') )
    ref_sol_dir_2 = ref_sol_dir + '%s_BigN_%d'%(reference_mesh_type, BigNval) + '/'
    data_dir_2 = data_dir + '%s_BigN_%d'%(reference_mesh_type, BigNval) + '/'
    
    for M in M_values: # data space discretization mesh
        Mval = M**(1 + (dim_input-1)*(data_discretization_type == 'reg') )
        ref_sol_dir_3 = ref_sol_dir_2 + '%s_M_%d'%(data_discretization_type, Mval) + '/'
        data_dir_3 = data_dir_2 + '%s_M_%d'%(data_discretization_type, Mval) + '/'
        print 'M = %d'%Mval
        for I in I_values: # integration mesh
            Ival = I**(1 + (dim_input-1)*(integration_mesh_type == 'reg') )
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
            
            print 'I = %d\n\n'%Ival 
            
            
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
                str1 += '%d'%Nval
                for j in range(len(QoI_choice_list)):
                    str1 += ' & %2.2e'%data_for_M[i][j]
                str1 += '\\\\ \n'
            str1 += '\\end{tabular}\n\\end{table}'
            print str1
            
            print '\n\n'
            new_data_filename = data_dir_3 + 'Plot-' + '%s_BigN_%d'%(reference_mesh_type, BigNval) + \
                    '_' + '%s_M_%d'%(data_discretization_type, Mval) + '_' + \
                    '%s_I_%d'%(integration_mesh_type, Ival) + '.eps'
            plt.cla()
            lines = []
            
            ref_vec = np.array([200, 6400])
            plt.plot(ref_vec,10./np.sqrt(ref_vec), linewidth=1.0, ls = '-', color = 'k')

            for qoi_idx in range(len(QoI_choice_list)):
                lines.append( plt.plot(N_values, data_for_M[:,qoi_idx], 'h', label = 'Q(%s)'%alpha[qoi_idx]) ) # NOTE not sure if label is working correctly
                plt.setp(lines[qoi_idx], linewidth=1.0,ls='--')
                if qoi_idx == 0:
                    plt.setp(lines, linewidth=1.0,ls='-')
                plt.setp(lines[qoi_idx], color = np.repeat(line_colors[qoi_idx],3,axis=0) )
            if recover:
                plt.title('Hellinger Distance with I = %d\n BigN = %d, M = %d'%(Ival, BigNval, Mval))
            else: 
                plt.title('Hellinger Distances (I = %d, BigN = %d) for the\nParameter i.d. Problem w/ rect_size = %s'%(Ival, BigNval, rect_size))
            plt.xlabel('Number of Samples', size='small')
            plt.ylabel('Hellinger Distance\n (%dE5 MC samples)'%(Ival/1E5), size='small')
            plt.xscale('log')
            plt.yscale('log')
            
            plt.legend(['MC Conv. Rate', '$Q^{(a)}$', '$Q^{(b)}$'], loc = 'lower left', fontsize = 'small') # NOTE: manually creating legend
            # plt.axis([20, 7500, 5E-3, 1])
            plt.savefig(new_data_filename)
            
            # 
            # for sol_num in range(len(QoI_choice_list)): # number of possible QoIs
            #     QoI_indices = QoI_choice_list[sol_num]
            #     
            #             
            # 
            #     for N in N_values:
            #         Nval = N**(1 + (dim_input-1)*(estimate_mesh_type == 'reg') )
            #         est_sol_dir_4 = est_sol_dir_3 + '%s_N_%d'%(estimate_mesh_type, Nval) + '/' # all trial sols inside this folder
            #         
            #         temp_array = np.zeros( (num_trials, len(QoI_choice_list)) )
            #         for trial in range(num_trials):
            #             est_sol_filename = est_sol_dir_4 + 'SolQoI_choice_%d'%(sol_num+1) + '-' + \
            #             '%s_M_%d'%(data_discretization_type, Mval) + \
            #             '_' + '%s_N_%d'%(estimate_mesh_type, Nval) + '_trial_%d'%(trial)
            #             
            #             est_ptr_filename = cwd + '/' + results_dir + '/' + sub_dirs[2] + '/' + \
            #                     '%s_N_%d'%(estimate_mesh_type, Nval) + '/' + \
            #                     'ptr_from_' + '%s_I_%d'%(integration_mesh_type, Ival) + \
            #                     '_to_' + '%s_N_%d'%(estimate_mesh_type, Nval) + '_trial_%d'%(trial) + '.npy'
            #                             
            #             Est_Disc = samp.load_discretization(est_sol_filename)
            #             est_ptr = np.load(est_ptr_filename)
            #             print 'Computing for I = %5d, BigN = %6d, M = %4d, N = %6d, trial # %3d'%(Ival, BigNval, Mval, Nval, trial)
            #             
            #             temp_array[trial, sol_num] = mc_Hellinger(Integration_Set,
            #                     Ref_Disc, ref_ptr, 
            #                     Est_Disc, est_ptr)
            #         data_dict[Nval]['data'] = temp_array
            #         data_dict[Nval]['stats'] = [ np.mean(temp_array, axis=0), np.var(temp_array, axis=0) ]           
            #         
            # # save data object
            # 
            # np.save(data_filename, data_dict)

