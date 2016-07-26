from new_estimatedist_funs import *

dim_input = 2
skew_range = [n+1 for n in range(3)]
dim_output = len(skew_range)

my_model = make_model(skew_range)
sampler = bsam.sampler(my_model)

dim_range_each = [0, 1]
dim_range = np.repeat([dim_range_each], dim_input, axis=0) # np.array([ [0, 1], [0, 1] ])
QoI_choice_list = [ [0, n+1] for n in range(dim_output)]

alpha = 1 # TODO add functionality for multiple alpha, betas - one per input_dim
beta = 1

# regular or random for all of them.
data_discretization_type = 'reg'
M_values = [2, 4, 5]
num_samples_emulate_data_space = 1E4

reference_mesh_type = 'reg'
BigN_values = [100]
# BigN_values = [1E5]

estimate_mesh_type = 'rand'
N_values = [25*2**n for n in range(4,10)]
# N_values = [4, 16, 25, 100, 400, 2500, 10000 ]
use_volumes = True # use calculateP.prob_with_emulated_volumes or just calculateP.prob - this uses emulated points
num_emulated_input_samples = 1E5

integration_mesh_type =  'rand'
I_values = [1E3, 1E4, 1E5] # map(int, [1E3, 1E4, 1E5]) 

num_trials = 50

cwd = os.getcwd()
results_dir = 'results'
sub_dirs = ['postprocess_data', 'integration_sets', 'est_discretizations', 'est_solutions', 'ref_solutions']
ref_sol_dir =  cwd + '/' + results_dir + '/' + sub_dirs[4] + '/' # commonly used
est_sol_dir =  cwd + '/' + results_dir + '/' + sub_dirs[3] + '/' # commonly used
data_dir = cwd + '/' + results_dir + '/' + sub_dirs[0] + '/'
