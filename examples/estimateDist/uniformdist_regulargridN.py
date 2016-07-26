# Copyright (C) 2014-2016 The BET Development Team

from estimatedist import *

rand_mult = 1 # random multiplier to test out sensitivity to randomness
# num_samples_param_space = 1E4 # this is N
# grid_cells_per_dim = 5 # Discretizing the Data Space using a regular grid on the input space - this is = log_dim (M)
# theta = 90   # degrees

# MC_assumption = True # (for your input samples)

save_ref_plot = False
save_disc = False
save_ref_disc = False

alpha = 1
beta = 1

num_sample_list = [4, 16, 25, 100, 400, 2500, 10000 ]
# num_sample_list = [25*2**n for n in range(5,10)]
# num_hell_bins_list = [2 ,4, 5, 10, 20, 50, 100]
# num_discr_list = [2, 3, 4, 6, 8, 9, 12]
num_hell_bins_list = [1E4]
num_discr_list = [2, 4, 5] 
num_trials = 1
H = {h:{ i:{ j:{} for j in num_sample_list} for i in num_discr_list } for h in num_hell_bins_list}
QoI_choice_list = [[0, 1]]
Nref_grid_per_dim = 100
skew_range = [n+1 for n in range(3)] # 1, 2, ..., 10
folder = 'HSN_test_fine'
emulate_option = True

def make_model(theta):
    # theta_rad = theta*2*(np.pi)/360
    def my_model(parameter_samples):
        Q_map = np.array([[1.0, 0.0], [np.cos(theta), np.sin(theta)]])
        # Q_map = np.array( [ [np.cos(theta_rad), -np.sin(theta_rad)], [np.sin(theta_rad), np.cos(theta_rad)] ] )
        # Q_map = np.array([[1.0, 0.0], [round(np.cos(theta),4), round(np.sin(theta),4)]])
        QoI_samples = np.dot(parameter_samples, np.transpose(Q_map))
        return QoI_samples
    return my_model

# int_s_sets = {h:[] for h in num_hell_bins_list} # integration discretization schemes
num_integration_points = num_hell_bins_list[0]
dim_input = 2
dim_range = [0, 1]
Integration_Sample_Set = samp.sample_set(dim_input)
Integration_Sample_Set.set_domain(np.repeat([dim_range], dim_input, axis=0))

# for num_hell_bins in num_hell_bins_list:
    # Integration_Sample_Set = bsam.regular_sample_set(Integration_Sample_Set, num_samples_per_dim = np.repeat(num_hell_bins, dim_input, axis=0) )
    # int_s_sets[num_hell_bins] = Integration_Sample_Set
    
Integration_Sample_Set = bsam.random_sample_set('random', Integration_Sample_Set, num_samples = num_integration_points)

for s in skew_range:
    
    my_model = make_model( np.arcsin(1./s) )

    for grid_cells_per_dim in num_discr_list:

        # print 'M = %4d'%(grid_cells_per_dim)
        
        Reference_Discretization, Partition_Set, Emulated_Set = \
            generate_N_reference(my_model, Nref_grid_per_dim, grid_cells_per_dim, alpha, beta, save_ref_disc, save_ref_plot)
        # ref_disc = {h:[] for h in num_hell_bins_list}
        # for num_hell_bins in num_hell_bins_list:
        #     ref_disc[num_hell_bins] = calculateP.prob_from_discretization_input(Reference_Discretization, int_s_sets[num_hell_bins])

        print 'References Done'
        for num_samples_param_space in num_sample_list:
            np.random.seed(num_samples_param_space*rand_mult)
            H_temp = {h: np.zeros((num_trials, len(QoI_choice_list))) for h in num_hell_bins_list}
            # rand_int = np.random.randint(num_trials) # print out one random recovered distribution
            for trial in range(num_trials):
                
                My_Discretization, Partition_Discretization, Emulated_Discretization = \
                    generate_model_discretizations(my_model, Partition_Set, Emulated_Set, num_samples_param_space, alpha, beta, emulate_option)
                    
                for qoi_choice_idx in range(len(QoI_choice_list)):
                    QoI_indices = QoI_choice_list[qoi_choice_idx]
                    
                    my_discretization = invert_using(My_Discretization, Partition_Discretization, Emulated_Discretization, QoI_indices, emulate_option)
                    
                    for num_hell_bins in num_hell_bins_list:
                        # H_temp[num_hell_bins][trial, qoi_choice_idx] = em_Hellinger(int_s_sets[num_hell_bins], Reference_Discretization, my_discretization)
                        H_temp[num_hell_bins][trial, qoi_choice_idx] = em_Hellinger(Integration_Sample_Set, Reference_Discretization, my_discretization)
                        
            # print '\t %d Trials for N = %4d samples completed.\n'%(num_trials, num_samples_param_space)
            for num_hell_bins in num_hell_bins_list: 
                H[num_hell_bins][grid_cells_per_dim][num_samples_param_space]['data'] = H_temp[num_hell_bins] # Hellinger distances as a list (each trial)
                H[num_hell_bins][grid_cells_per_dim][num_samples_param_space]['stats'] = [np.mean(H_temp[num_hell_bins], axis=0), np.var(H_temp[num_hell_bins], axis=0)]            
            print '\t', 'mean for Skew = %d, N = %4d:'%(s, num_samples_param_space), [ H[num_hell_bins][grid_cells_per_dim][num_samples_param_space]['stats'][0] for num_hell_bins in num_hell_bins_list]
                # print '\t', 'var:', H[grid_cells_per_dim][num_samples_param_space]['stats'][1]
        
    np.save('%s/(%d,%d)_skew_%d.npy'%(folder, alpha, beta, s), H)
