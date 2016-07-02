# Copyright (C) 2014-2016 The BET Development Team

from estimatedist import *
from matplotlib import pyplot as plt

rand_mult = 1 # random multiplier to test out sensitivity to randomness
# num_samples_param_space = 1E4 # this is N
# grid_cells_per_dim = 5 # Discretizing the Data Space using a regular grid on the input space - this is = log_dim (M)
save_plots = False
save_ref_plot = False
save_disc = False
save_ref_disc = False
MC_assumption = True # (for your input samples)
alpha = 20
beta = 20

num_sample_list = [25*2**n for n in range(9)]
max_grid = 10
num_discr_list = range(3,max_grid+1,1)
num_trials = 3
H = { i:{ j:{} for j in num_sample_list} for i in num_discr_list }
QoI_choice_list = [[0, 1]]

theta = 90   # degrees
def make_model(theta):
    theta_rad = theta*2*(np.pi)/360
    def my_model(parameter_samples):
        Q_map = np.array([[1.0, 0.0], [np.cos(theta_rad), np.sin(theta_rad)]])
        # Q_map = np.array([[1.0, 0.0], [round(np.cos(theta),4), round(np.sin(theta),4)]])
        QoI_samples = np.dot(parameter_samples, np.transpose(Q_map))
        return QoI_samples
    return my_model

theta_range = np.linspace(0,90,15)

for theta in theta_range:
    
    my_model = make_model(theta)

    for grid_cells_per_dim in num_discr_list:

        # print 'M = %4d'%(grid_cells_per_dim)
        
        Reference_Discretization, Partition_Set, Emulated_Set = generate_reference(grid_cells_per_dim, alpha, beta, save_ref_disc, save_ref_plot)
        
        (_, ref_marginal) = plotP.calculate_2D_marginal_probs(Reference_Discretization._input_sample_set, nbins = grid_cells_per_dim)
        
        for num_samples_param_space in num_sample_list:
            # np.random.seed(num_samples_param_space*rand_mult)
            H_temp = np.zeros((num_trials, len(QoI_choice_list)))
            # rand_int = np.random.randint(num_trials) # print out one random recovered distribution
            for trial in range(num_trials):
                
                My_Discretization, Partition_Discretization, Emulated_Discretization = \
                    generate_model_discretizations(my_model, Partition_Set, Emulated_Set, num_samples_param_space, alpha, beta)
                    
                for qoi_choice_idx in range(len(QoI_choice_list)):
                    QoI_indices = QoI_choice_list[qoi_choice_idx]
                    my_discretization = invert_using(My_Discretization, Partition_Discretization, Emulated_Discretization, QoI_indices)
                    
                    (bins, temp_marginal) = plotP.calculate_2D_marginal_probs(my_discretization._input_sample_set, nbins = grid_cells_per_dim)
                    if save_plots == True: # save only first random recovered distribution
                        plotP.plot_2D_marginal_probs(temp_marginal, bins, Reference_Discretization._input_sample_set, 
                                    filename = "1_(%d,%d)_M%d_N%d_Recovered_Distribution%d"%(alpha, beta, grid_cells_per_dim, num_samples_param_space, qoi_choice_idx), 
                                    file_extension = ".png", plot_surface=False)
                    H_temp[trial, qoi_choice_idx] = Hellinger(ref_marginal[(0,1)], temp_marginal[(0,1)])
            # print '\t %d Trials for N = %4d samples completed.\n'%(num_trials, num_samples_param_space)
            H[grid_cells_per_dim][num_samples_param_space]['data'] = H_temp
            H[grid_cells_per_dim][num_samples_param_space]['stats'] = [np.mean(H_temp, axis=0), np.var(H_temp, axis=0)]            
            print '\t', 'mean for N = %4d:'%num_samples_param_space, H[grid_cells_per_dim][num_samples_param_space]['stats'][0]
            # print '\t', 'var:', H[grid_cells_per_dim][num_samples_param_space]['stats'][1]
    np.save('dict_results_theta_%d.npy'%theta,H)

# print H
'''
plt.cla()
lines = plt.plot(num_sample_list, H_mean[:,0], 'k--', num_sample_list, H_mean[:,1],'b:')
plt.setp(lines, 'linewidth', 2.0)
plt.xlabel('N')
plt.ylabel('H')
plt.xscale('log')
plt.yscale('log')
# plt.axis([20, 2000, 1E-3, 1])
plt.title( 'Mean Hellinger Distance (%3d Trials ) as a \n Function of N for a Fixed M = %d'%(num_trials, grid_cells_per_dim**2 ) )
# plt.show()
plt.savefig('(%d,%d)_Mean_Comparison_M%d.png'%(alpha, beta, grid_cells_per_dim**2) )

H_var = np.array(H_var)
plt.cla()
lines = plt.plot(num_sample_list, H_var[:,0], 'k--', num_sample_list, H_var[:,1],'b:')
plt.setp(lines, 'linewidth', 2.0)
plt.xlabel('N')
plt.ylabel('Variance')
plt.xscale('log')
# plt.yscale('log')
# plt.axis([20, 2000, 1E-8, 1E-1])
plt.title( 'Variance in Hellinger Distances (%3d Trials ) as a \n Function of N for a Fixed M = %d'%(num_trials, grid_cells_per_dim**2 ) )
# plt.show()
plt.savefig('(%d,%d)_Variance_Comparison_M%d.png'%(alpha, beta, grid_cells_per_dim**2) )
'''
# plt.close('all')