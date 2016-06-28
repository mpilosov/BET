# Copyright (C) 2014-2016 The BET Development Team

from estimatedist import *
from matplotlib import pyplot as plt
np.random.seed(50)


# num_samples_param_space = 1E4 # this is N
grid_cells_per_dim = 5 # Discretizing the Data Space using a regular grid on the input space - this is = log_dim (M)
plotting_on = False
save_disc = False
MC_assumption = True # (for your input samples)
H_L = []
num_sample_list = [100*2**n for n in range(6)]
for num_samples_param_space in num_sample_list:
    H_L.append(generate_data(num_samples_param_space, grid_cells_per_dim))
H_L = np.array(H_L)
plt.plot(num_sample_list, H_L[:,0], num_sample_list, H_L[:,1])
plt.xlabel('N')
plt.ylabel('H')
plt.xscale('log')
plt.yscale('log')
plt.title( 'Plot for M = %d'%grid_cells_per_dim**2 )
plt.show()