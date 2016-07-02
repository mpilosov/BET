from matplotlib import pyplot as plt
import numpy as np

alpha = 1
beta = 1
num_trials = 50 # for each N
theta = 90
H = np.load('dict_comp_results_theta_%d.npy'%theta)
N_vec = [25*2**n for n in range(9)] # should save these into data file later on.
max_grid = 10
M_vec= range(3,max_grid+1,1)

H_mean = np.array([ [H[M][N]['stats'][0] for N in N_vec] for M in M_vec]) # M down rows, N down columns
H_var = np.array([ [H[M][N]['stats'][1] for N in N_vec] for M in M_vec]) # M down rows, N down columns

var_or_mean = 'mean'
for M in M_vec:
    plt.cla()
    if var_or_mean == 'mean':
        lines = plt.plot(N_vec, H_mean[M,:], 'k--', N_vec, H_mean[M,:],'b:')
        plt.title( 'Mean Hellinger Distance (%3d Trials ) as a \n Function of N for a Fixed M = %d'%(num_trials, M**2 ) )
    else:
        lines = plt.plot(N_vec, H_var[M,:], 'k--', N_vec, H_var[M,:],'b:')
        plt.title( 'Variance in Hellinger Distances (%3d Trials ) as a \n Function of N for a Fixed M = %d'%(num_trials, M**2 ) )

    plt.setp(lines, 'linewidth', 2.0)
    plt.xlabel('N')
    plt.ylabel('H')
    plt.xscale('log')
    plt.yscale('log')
    # plt.axis([20, 2000, 1E-3, 1])
    # plt.show()
    plt.savefig('(%d,%d)_Mean_Comparison_M%d.png'%(alpha, beta, M**2) )

plt.close('all')