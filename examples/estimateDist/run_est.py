# Copyright (C) 2014-2016 The BET Development Team

import estimatedist

np.random.seed(20)


num_samples_param_space = 1E4 # this is N
grid_cells_per_dim = 10 # Discretizing the Data Space using a regular grid on the input space - this is = log_dim (M)
plotting_on = False
save_disc = False
MC_assumption = True # (for your input samples)
H_L = []
for num_samples_param_space in [100*2**n for n in range(5)]:
    H_L.append([num_samples_param_space, generate_data(num_samples_param_space, grid_cells_per_dim)])
print H_L