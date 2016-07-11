# Copyright (C) 2014-2016 The BET Development Team

from estimatedist import *
from matplotlib import pyplot as plt

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

num_sample_list = [25*2**n for n in range(10)]
# max_grid = 6
num_hell_bins_list = [2, 3]
# num_discr_list = range(3,max_grid+1,1)
num_discr_list = [2, 3, 4, 6, 8, 9, 12] 
num_trials = 50
H = {h:{ i:{ j:{} for j in num_sample_list} for i in num_discr_list } for h in num_hell_bins_list}
QoI_choice_list = [[0, 1]]

def make_model(theta):
    theta_rad = theta*2*(np.pi)/360
    def my_model(parameter_samples):
        Q_map = np.array([[1.0, 0.0], [np.cos(theta_rad), np.sin(theta_rad)]])
        # Q_map = np.array( [ [np.cos(theta_rad), -np.sin(theta_rad)], [np.sin(theta_rad), np.cos(theta_rad)] ] )
        # Q_map = np.array([[1.0, 0.0], [round(np.cos(theta),4), round(np.sin(theta),4)]])
        QoI_samples = np.dot(parameter_samples, np.transpose(Q_map))
        return QoI_samples
    return my_model

# H2s folder:
skew_range = [n+1 for n in range(10)] # 1, 2, ... 10
theta_range_rad = [ np.arcsin(1./s) for s in skew_range ]
theta_range_deg = [t*360/(2*np.pi) for t in theta_range_rad ]
# H2 folder:
# theta_range = [int(i) for i in np.floor( np.linspace(0,90,15)[:-1] )] + range(84,91)

for theta in theta_range_deg:
    my_model = make_model(theta)

    for grid_cells_per_dim in num_discr_list:

        # print 'M = %4d'%(grid_cells_per_dim)
        
        Reference_Discretization, Partition_Set, Emulated_Set = generate_reference(grid_cells_per_dim, alpha, beta, save_ref_disc, save_ref_plot)
        ref_marginal = {h:[] for h in num_hell_bins_list}
        for num_hell_bins in num_hell_bins_list:
            (_, temp_ref_marginal) = plotP.calculate_2D_marginal_probs(Reference_Discretization._input_sample_set, nbins = num_hell_bins)
            ref_marginal[num_hell_bins] = temp_ref_marginal
            
        for num_samples_param_space in num_sample_list:
            np.random.seed(num_samples_param_space*rand_mult)
            H_temp = {h: np.zeros((num_trials, len(QoI_choice_list))) for h in num_hell_bins_list}
            # rand_int = np.random.randint(num_trials) # print out one random recovered distribution
            for trial in range(num_trials):
                
                My_Discretization, Partition_Discretization, Emulated_Discretization = \
                    generate_model_discretizations(my_model, Partition_Set, Emulated_Set, num_samples_param_space, alpha, beta)
                    
                for qoi_choice_idx in range(len(QoI_choice_list)):
                    QoI_indices = QoI_choice_list[qoi_choice_idx]
                    
                    my_discretization = invert_using(My_Discretization, Partition_Discretization, Emulated_Discretization, QoI_indices)
                    
                    for num_hell_bins in num_hell_bins_list:
                        (bins, temp_marginal) = plotP.calculate_2D_marginal_probs(my_discretization._input_sample_set, nbins = num_hell_bins)
                        H_temp[num_hell_bins][trial, qoi_choice_idx] = Hellinger(ref_marginal[num_hell_bins][(0,1)], temp_marginal[(0,1)])
            # print '\t %d Trials for N = %4d samples completed.\n'%(num_trials, num_samples_param_space)
            for num_hell_bins in num_hell_bins_list: 
                H[num_hell_bins][grid_cells_per_dim][num_samples_param_space]['data'] = H_temp[num_hell_bins] # Hellinger distances as a list (each trial)
                H[num_hell_bins][grid_cells_per_dim][num_samples_param_space]['stats'] = [np.mean(H_temp[num_hell_bins], axis=0), np.var(H_temp[num_hell_bins], axis=0)]            
            print '\t', 'mean for Theta = %d, N = %4d:'%(theta, num_samples_param_space), [ H[num_hell_bins][grid_cells_per_dim][num_samples_param_space]['stats'][0] for num_hell_bins in num_hell_bins_list]
                # print '\t', 'var:', H[grid_cells_per_dim][num_samples_param_space]['stats'][1]
        
    np.save('HS/(%d,%d)_theta_%d.npy'%(alpha, beta, theta), H)
