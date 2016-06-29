# Copyright (C) 2014-2016 The BET Development Team

from estimatedist import *
from matplotlib import pyplot as plt

rand_mult = 1 # random multiplier to test out sensitivity to randomness
# num_samples_param_space = 1E4 # this is N
# grid_cells_per_dim = 5 # Discretizing the Data Space using a regular grid on the input space - this is = log_dim (M)
plotting_on = False
save_disc = False
MC_assumption = True # (for your input samples)
alpha = 1
beta = 1

num_sample_list = [25*2**n for n in range(7)]
# num_sample_list = [40]
num_trials = 20
max_grid = 3

for grid_cells_per_dim in [3]: #range(3,max_grid+1,2):
    H_mean = []
    H_var = []
    print 'M = %4d'%(grid_cells_per_dim**2)
    for num_samples_param_space in num_sample_list:
        np.random.seed(num_samples_param_space*rand_mult)
        H_temp = []
        for trial in range(num_trials):
            H_temp.append( generate_data(num_samples_param_space, grid_cells_per_dim, alpha, beta, plotting_on = plotting_on) )
        print '\t %d Trials for N = %4d samples completed.\n'%(num_trials, num_samples_param_space)
        H_temp = np.array(H_temp)
        H_mean.append( [ np.mean(H_temp[:,0]), np.mean(H_temp[:,1]) ] )
        H_var.append( [ np.var(H_temp[:,0]), np.var(H_temp[:,1]) ] )
    H_mean = np.array(H_mean)
    print H_mean
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