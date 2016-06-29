
import numpy as np
import bet.postProcess as postProcess
import bet.calculateP.simpleFunP as simpleFunP
import bet.calculateP.calculateP as calculateP
import bet.postProcess.plotP as plotP
import bet.postProcess.plotDomains as plotD
import bet.postProcess.postTools as postTools
import bet.sample as samp
import bet.sampling.basicSampling as bsam
import bet.sensitivity.gradients as grad
import bet.sensitivity.chooseQoIs as cqoi
import bet.Comm as comm
np.random.seed(40)
# nbins = [3, 3]
# bins = []
# for i in range(2):
#     bins.append(np.linspace(0, 1, nbins[i]+1))
# for i in range(2):
#     for j in range(2):
#         (marg, _) = np.histogramdd(np.array( [[ 0.54395929,  0.06344683],
#                                                [ 0.96550781,  0.72614244],
#                                                [ 0.41163848,  0.18906439],
#                                                [ 0.13182463,  0.59977952],
#                                                [ 0.59596785,  0.25229687],
#                                                [ 0.57608341,  0.78337879],
#                                                [ 0.11564915,  0.8291189 ],
#                                                [ 0.53665273,  0.98189679],
#                                                [ 0.91618493,  0.80821008],
#                                                [ 0.23902877,  0.90631674],
#                                                [ 0.80121622,  0.11964556],
#                                                [ 0.95134873,  0.37588735],
#                                                [ 0.19827056,  0.22715624],
#                                                [ 0.89165566,  0.44543174],
#                                                [ 0.65164622,  0.67592569],
#                                                [ 0.13164588,  0.10537085],
#                                                [ 0.28503375,  0.64442602],
#                                                [ 0.11810532,  0.35029385],
#                                                [ 0.78927627,  0.05953596],
#                                                [ 0.76864736,  0.37488839],
#                                                [ 0.97625284,  0.00578395],
#                                                [ 0.13224108,  0.44405854],
#                                                [ 0.42520129,  0.48696857],
#                                                [ 0.14930928,  0.61975191],
#                                                [ 0.47074413,  0.27556827],
#                                                [ 0.44273567,  0.16316201],
#                                                [ 0.7853008 ,  0.09927963],
#                                                [ 0.47610039,  0.38573056],
#                                                [ 0.45514064,  0.48583119],
#                                                [ 0.2126122 ,  0.4946755 ],
#                                                [ 0.87695948,  0.25696185],
#                                                [ 0.99241028,  0.22323727],
#                                                [ 0.90143702,  0.81490299],
#                                                [ 0.57713527,  0.96313962],
#                                                [ 0.69044652,  0.94002192],
#                                                [ 0.7420612 ,  0.01364398],
#                                                [ 0.89053773,  0.0380261 ],
#                                                [ 0.12953605,  0.54223007],
#                                                [ 0.65217515,  0.25469464],
#                                                [ 0.18744806,  0.63908146]] ) ,
#                 bins=[bins[i], bins[j]],
#                 weights= 
#                 [ 0.02080808,  0.02691358,  0.02080808,  0.05222222,  0.03888889,
#                 0.02691358,  0.02691358,  0.02691358,  0.02691358,  0.02691358,
#                 0.02080808,  0.01746032,  0.02080808,  0.01111111,  0.0362963 ,
#                 0.02080808,  0.0362963 ,  0.02111111,  0.02080808,  0.01746032,
#                 0.02080808,  0.01746032,  0.01746032,  0.05222222,  0.02111111,
#                 0.02080808,  0.02080808,  0.01746032,  0.01746032,  0.01746032,
#                 0.02111111,  0.02111111,  0.02691358,  0.02691358,  0.02691358,
#                 0.02080808,  0.02080808,  0.01111111,  0.02111111,  0.0362963 ] )
#         marg = np.ascontiguousarray(marg)
#         marg_temp = np.copy(marg)
#         marginals[(i, j)] = marg_temp
# print marginals

def identity_model(parameter_samples):
    Q_map = np.array([[1.0, 0.0], [0.0, 1.0]])
    QoI_samples = np.dot(parameter_samples, np.transpose(Q_map))
    return QoI_samples

num_samples_param_space = 40
dim_input = 2
grid_cells_per_dim = 3 # discretization of data space (regular grid)
dim_range = [0.0, 1.0]
alpha = 1
beta = 1
num_samples_emulate_data_space = (grid_cells_per_dim**dim_input)*100

eye = bsam.sampler(identity_model)

# create partition and emulation sets
Partition_Set = samp.sample_set(dim_input)
Partition_Set.set_domain(np.repeat([dim_range], dim_input, axis=0))
Partition_Set = bsam.regular_sample_set(Partition_Set, num_samples_per_dim = np.repeat(grid_cells_per_dim, dim_input, axis=0))

Emulated_Set = samp.sample_set(dim_input)
Emulated_Set.set_domain(np.repeat([dim_range], dim_input, axis=0))
# Emulation from true distribution 
Emulated_Set.set_values(np.array( np.transpose([ np.random.beta(a=alpha, b=beta,
            size=num_samples_emulate_data_space) for i in range(dim_input) ]) ))


Reference_Set = samp.sample_set(dim_input)
Reference_Set.set_domain(np.repeat([dim_range], dim_input, axis=0))
Reference_Set = bsam.regular_sample_set(Reference_Set, num_samples_per_dim = np.repeat(grid_cells_per_dim, dim_input, axis=0))

Reference_Discretization = eye.compute_QoI_and_create_discretization(Reference_Set)
# Reference_Discretization._input_sample_set.estimate_volume_mc() # The MC assumption is true.
Reference_Emulation = eye.compute_QoI_and_create_discretization(Emulated_Set)


simpleFunP.user_partition_user_distribution(Reference_Discretization, 
                                            Reference_Discretization, 
                                            Reference_Emulation)

# set probabilities - the map was the identity, so the inverse problem is trivial
Reference_Discretization._input_sample_set.set_probabilities(Reference_Discretization._output_probability_set._probabilities)

# compute reference marginal
(bins, ref_marginals2D) = plotP.calculate_2D_marginal_probs(Reference_Discretization._input_sample_set, nbins = grid_cells_per_dim)
print 'Reference Marginal:'
print ref_marginals2D[(0,1)]

print '\nReference Probabilities:'
print Reference_Discretization._input_sample_set._probabilities



Input_Samples = samp.sample_set(dim_input)
Input_Samples.set_domain(np.repeat([dim_range], dim_input, axis=0))
Input_Samples = bsam.random_sample_set('random', Input_Samples, num_samples = num_samples_param_space)
# Input_Samples = bsam.regular_sample_set(Input_Samples, num_samples_per_dim = np.repeat(grid_cells_per_dim, dim_input, axis=0))
Input_Samples.estimate_volume_mc()

# Solve Inverse Problem
My_Discretization = eye.compute_QoI_and_create_discretization(Input_Samples)
Partition_Discretization = eye.compute_QoI_and_create_discretization(Partition_Set)
Emulated_Discretization = eye.compute_QoI_and_create_discretization(Emulated_Set)

simpleFunP.user_partition_user_distribution(My_Discretization,
                                            Partition_Discretization,
                                            Emulated_Discretization)
calculateP.prob(My_Discretization)                                            

print '\nInverse Problem, Probabilities:'
print My_Discretization._input_sample_set._probabilities_local
print 'Pointer:'
print My_Discretization._io_ptr_local
print '\nInvere Solution Marginal'
# print marginals for comparison
(bins, marginals2D) = plotP.calculate_2D_marginal_probs(My_Discretization._input_sample_set, nbins = grid_cells_per_dim)
print marginals2D[(0, 1)]
print '\n'