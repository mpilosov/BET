from matplotlib import pyplot as plt
import numpy as np

alpha = 1
beta = 1
num_trials = 50 
var_or_mean = 'mean'
max_grid = 10
M_vec= range(3,max_grid+1,1)
N_vec = [25*2**n for n in range(9)]
# choose thetas to compare. can also just fix a theta value to get an individual line.
# theta_range = [int(i) for i in np.floor( np.linspace(0,90,15)[:-1] )] + range(84,91)
# theta_range = [0, 6, 12, 19, 25, 32, 38, 45, 51, 57, 64, 70, 77, 83]
# theta_range = range(83,91)
# theta_range = [0, 25, 45] # to demo that orthogonal maps are independent of rotation 
# theta_range = [0, 22, 45, 67, 90] # to demo that a 45deg skew map rotated around shows symmetry
# theta_range = [0, 22, 45] # just the distinct ones
# theta_range = [0, 5, 10, 15, 20, 22, 25] 
theta_range = [0, 10, 15, 20, 25]

# theta_range = [22,67] # symmetric pairs shown here
# theta_range = [0, 90] # and here. so something around 22 is the minimizer.

line_colors = np.linspace(0.8, 0, len(theta_range)) # LIGHT TO DARK - LOW to HIGH Theta

for M_idx in range(len(M_vec)):
    M = M_vec[M_idx]
    plt.cla()
    for theta_idx in range( len(theta_range)) :
        theta = theta_range[theta_idx]
        # H = np.load('base/(%d,%d)_dict_results_theta_%d.npy'%(alpha, beta, theta)) # fixed [1, 0], rotating second vector until orthogonal (resolution = 15)
        # H = np.load('rot/(%d,%d)_dict_orth_results_theta_%d.npy'%(alpha, beta, theta) ) # orthogonal map rotated through 0 .. 90 (resolution = 15)
        H = np.load('rot/(%d,%d)_dict_skew_results_theta_%d.npy'%(alpha, beta, theta) ) # 45 deg skewed map rotated 0 .. 90 (resolution = 5)
        H = H.item()
        H_mean = np.array([ [ H[MM][NN]['stats'][0] for NN in N_vec] for MM in M_vec]) # M down rows, N across columns
        H_var = np.array([ [ H[MM][NN]['stats'][1] for NN in N_vec] for MM in M_vec]) # M down rows, N across columns
        
        # plt.cla()
        if var_or_mean == 'mean':
            for i in range(len(H_mean[M_idx,:][0])):
                lines = plt.plot(N_vec, [H_mean[M_idx, N_idx][i] for N_idx in range(len(N_vec)) ], 'h')
            plt.title( 'Mean Hellinger Distance (%3d Trials ) as a \n Function of N for a Fixed M = %d'%(num_trials, M**2 ) )
        else:
            for i in range(len(H_var[M_idx,:][0])):
                lines = plt.plot(N_vec, [H_var[M_idx, N_idx][i] for N_idx in range(len(N_vec)) ], 'h')
            plt.title( 'Variance in Hellinger Distances (%3d Trials ) as a \n Function of N for a Fixed M = %d'%(num_trials, M**2 ) )

        plt.setp(lines, linewidth=1.0,ls='--')
        plt.setp(lines, color = np.repeat(line_colors[theta_idx],3,axis=0) )
    plt.xlabel('N')
    plt.ylabel('H')
    plt.xscale('log')
    plt.yscale('log')
    # plt.axis([20, 7500, 1E-3, 1])
    # plt.axis([20, 7500, 1E-3, 1])
    # plt.show()
    if var_or_mean == 'mean':
        # plt.axis([20, 7500, 1E-3, 1]) # the usual
        plt.axis([20, 7500, 1E-2, 1]) # good for (1,1)
        plt.savefig('rot/Skew_(%d,%d)_Mean_M%d.png'%(alpha, beta, M**2) )
    else:
        plt.axis([20, 7500, 1E-6, 1E-2])
        # plt.axis([20, 7500, 1E-18, 1E-15])
        plt.savefig('rot/Skew_(%d,%d)_Var_M%d.png'%(alpha, beta, M**2) )
            