from matplotlib import pyplot as plt
import numpy as np

alpha = 1
beta = 1
num_trials = 50 # for each N
theta = 0
H = np.load('dict_results_theta_%d.npy'%theta)
H = H.item()
N_vec = [25*2**n for n in range(9)] # should save these into data file later on.
max_grid = 10
M_vec= range(3,max_grid+1,1)

H_mean = np.array([ [ H[M][N]['stats'][0] for N in N_vec] for M in M_vec]) # M down rows, N down columns
H_var = np.array([ [ H[M][N]['stats'][1] for N in N_vec] for M in M_vec]) # M down rows, N down columns

for var_or_mean in ['mean', 'var']:
    for M_idx in range(len(M_vec)):
        M = M_vec[M_idx]
        plt.cla()
        if var_or_mean == 'mean':
            for i in range(len(H_mean[M_idx,:][0])):
                lines = plt.plot(N_vec, [H_mean[M_idx, N_idx][i] for N_idx in range(len(N_vec)) ], 'k--')
            plt.title( 'Mean Hellinger Distance (%3d Trials ) as a \n Function of N for a Fixed M = %d'%(num_trials, M**2 ) )
        else:
            for i in range(len(H_var[M_idx,:][0])):
                lines = plt.plot(N_vec, [H_var[M_idx, N_idx][i] for N_idx in range(len(N_vec)) ], 'k--')
            plt.title( 'Variance in Hellinger Distances (%3d Trials ) as a \n Function of N for a Fixed M = %d'%(num_trials, M**2 ) )

        plt.setp(lines, 'linewidth', 2.0)
        plt.xlabel('N')
        plt.ylabel('H')
        plt.xscale('log')
        plt.yscale('log')
        # plt.axis([20, 2000, 1E-3, 1])
        # plt.show()
        if var_or_mean == 'mean':
            plt.savefig('(%d,%d)_Theta_%d_Mean__M%d.png'%(alpha, beta, theta, M**2) )
        else:
            plt.savefig('(%d,%d)_Theta_%d_Variance__M%d.png'%(alpha, beta, theta, M**2) )

    for M_idx in range(len(M_vec)):
        M = M_vec[M_idx]
        # plt.cla()
        if var_or_mean == 'mean':
            lines = plt.plot(N_vec, H_mean[M_idx,:], '--')
            plt.title( 'Mean Hellinger Distance (%3d Trials ) as a \n Function of N for a Range of M = %d, %d, ... %d'%(num_trials, M_vec[0]**2, M_vec[1]**2, M_vec[-1]**2  ) )
        else:
            lines = plt.plot(N_vec, H_var[M_idx,:], '--')
            plt.title( 'Variance in Hellinger Distances (%3d Trials ) as a \n Function of N for a Range of M = %d, %d, ... %d'%(num_trials, M_vec[0]**2, M_vec[1]**2, M_vec[-1]**2  ) )

        plt.setp(lines, 'linewidth', 2.0)
        plt.xlabel('N')
        plt.ylabel('H')
        plt.xscale('log')
        plt.yscale('log')
        # plt.axis([20, 2000, 1E-3, 1])
        # plt.show()
    if var_or_mean == 'mean':
        plt.savefig('(%d,%d)_Theta_%d_Mean_Comparison_M_all.png'%(alpha, beta, theta) )
    else:
        plt.savefig('(%d,%d)_Theta_%d_Variance_Comparison_M_all.png'%(alpha, beta, theta) )
    plt.close('all')