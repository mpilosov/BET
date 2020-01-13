from estimatedist_funs import *
import pyprind



dim_input = 2
dim_output = 2

dim_range = np.array([[0.01, 0.2], [0.01, 0.2]])  # parameter space domain
# temp_locs_list = [0.125, 0.9, 0.7] # list of QoI locations
temp_locs_list = [0.25, 0.51, 0.67, 0.98]
QoI_choice_list = [[0, 1], [2, 3]]
# QoI_choice_list = [ [0, n] for n in range(1,len(temp_locs_list))] # first location fixed.

my_model = make_model(temp_locs_list)
sampler = bsam.sampler(my_model)

# emulation set is given these and that's it. Shouldn't bother with anything except uniform, since it's used for volumes.
alpha = 1  # TODO add functionality for multiple alpha, betas - one per input_dim
beta = 1

# regular or random for all of them.
data_discretization_type = 'reg'
M_values = [1]
# TODO append to Emulation_Discretization filename
num_samples_emulate_data_space = 1E5

reference_mesh_type = 'reg'
BigN_values = [100]

estimate_mesh_type = 'rand'
# N_values = [20*2**n for n in range(8)]
N_values = [100, 500, 1000, 2500, 5000]
# use calculateP.prob_with_emulated_volumes or just calculateP.prob - this uses emulated points
use_volumes = False
num_emulated_input_samples = 1E6

integration_mesh_type = 'rand'
I_values = [1E5]
num_trials = 10

# different reference kappas: regular grid in center of space.
# only one is chosen (ref_input_num) for simulation.
kappa_lin = np.linspace(0.01, 0.2, 5)[1:-1]  
# like meshgrid, only easier to parse
kappa_ref_locs = np.array([[k1, k2] for k1 in kappa_lin for k2 in kappa_lin])

# should be able to change this and run.sh
ref_input_num = 2  # 4 is middle. choose 0-8
ref_input = np.array([kappa_ref_locs[ref_input_num]])
Qref = my_model(ref_input)[0]
rect = 0.1  # currently scale, not size

recover = False

show_title = False
label_fsize = 20
tick_fsize = 14
legend_fsize = 14

# Initial Run
create_int_sets = True
create_data_discs = True
compute_emulated_set = False

create_ref_disc = True
compute_ref_sol = True

create_est_discs = True
compute_est_sol = True

# need to run build_pointers before uncommenting next section.

# ## Post-Processing - change description of uncertainty on output
# create_int_sets = False
# create_data_discs = False
# compute_emulated_set = False

# create_ref_disc = False
# compute_ref_sol = True

# create_est_discs = False
# compute_est_sol = True

cwd = os.getcwd()
results_dir = 'results_heatrod_3'
sub_dirs = ['postprocess_rectscale_%d' % (
    1+ref_input_num), 'integration_sets', 'est_discretizations', 'est_solutions', 'ref_solutions']
ref_sol_dir = cwd + '/' + results_dir + \
    '/' + sub_dirs[4] + '/'  # commonly used
est_sol_dir = cwd + '/' + results_dir + \
    '/' + sub_dirs[3] + '/'  # commonly used
data_dir = cwd + '/' + results_dir + '/' + sub_dirs[0] + '/'


