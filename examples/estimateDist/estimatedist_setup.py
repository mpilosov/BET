from estimatedist_funs import *
import pyprind

dim_input = 2
# skew_range = [n+1 for n in range(2)]
# skew_range = [1,2,4]
dim_output = 2

my_model = make_model(skew_range)
sampler = bsam.sampler(my_model)

dim_range_each = [0, 1]
dim_range = np.repeat([dim_range_each], dim_input, axis=0) # np.array([ [0, 1], [0, 1] ])
QoI_choice_list = [ [0, n+1] for n in range(len(skew_range))]

alpha = 1 # TODO add functionality for multiple alpha, betas - one per input_dim
beta = 1

# regular or random for all of them.
data_discretization_type = 'reg'
M_values = [1, 2, 3]
num_samples_emulate_data_space = 1E5 # TODO append to Emulation_Discretization file

reference_mesh_type = 'reg'
BigN_values = [200]
# BigN_values = [1E5]

estimate_mesh_type = 'rand'
# N_values = [2,5,10,20]
# N_values = [50, 200, 800, 3200, 6400]
N_values = [25*2**n for n in range(3,9)]
# N_values = [4, 16, 25, 100, 400, 2500, 10000 ]
use_volumes = False # use calculateP.prob_with_emulated_volumes or just calculateP.prob - this uses emulated points
num_emulated_input_samples = 1E6

integration_mesh_type =  'rand'
# I_values = [1E4] # map(int, [1E3, 1E4, 1E5]) 
I_values = [1E5]
num_trials = 50

ref_input = 0.5*np.ones(dim_input)
Qref =  my_model(ref_input)
rect_size = np.power(0.1, 1./dim_output) # make box with this sidelength 

recover = False

show_title = False
label_fsize = 20
tick_fsize = 14
legend_fsize = 14

## Initial Run
# create_int_sets = True
# create_data_discs = False
# compute_emulated_set = False
# create_ref_disc = True
# create_est_discs = True
# compute_ref_sol = True
# compute_est_sol = True

## Post-Processing - change description of uncertainty on output
create_int_sets = False
create_data_discs = False
compute_emulated_set = False
create_ref_disc = False
create_est_discs = False
compute_ref_sol = True
compute_est_sol = True

cwd = os.getcwd()
results_dir = 'results_IP_ref2d'
sub_dirs = ['postprocess_rect_01', 'integration_sets', 'est_discretizations', 'est_solutions', 'ref_solutions']
ref_sol_dir =  cwd + '/' + results_dir + '/' + sub_dirs[4] + '/' # commonly used
est_sol_dir =  cwd + '/' + results_dir + '/' + sub_dirs[3] + '/' # commonly used
data_dir = cwd + '/' + results_dir + '/' + sub_dirs[0] + '/'
