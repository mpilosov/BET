import scipy.io as sio
from extractqoivals import filename, temp_locs
import bet.sample as samp
import bet.sampling.basicSampling as bsam
import bet.calculateP.simpleFunP as simpleFunP
import bet.calculateP.calculateP as calculateP
import bet.postProcess.plotP as plotP
from simulation_2kappas_setup import * # imports numpy, sys, dolfin
from heatROD import *

QoI_choice_list = [ [0, n] for n in range(1,len(temp_locs))] # first location fixed.
kappa_lin = np.linspace(0.01, 0.2, 5)[1:-1]
kappa_ref_locs = [[k1,k2] for k1 in kappa_lin for k2 in kappa_lin] # like meshgrid, only easier to parse
rect_scale = 0.1 # percentage uncertainty per QoI parameter
dim_input = 2
dim_range = np.array([[0.01, 0.2],[0.01, 0.2]])
N_values = [20*2**n for n in range(0,7)]

matfile = sio.loadmat(filename)
loaded_input_samples = matfile['samples']
loaded_output_samples = matfile['data']


# Initialize the necessary sample objects - FULL DATAFRAME
input_samples = samp.sample_set(dim_input)
input_samples.set_domain( dim_range )
output_samples = samp.sample_set( len(temp_locs) )

# Set the input sample values from the imported file
input_samples.set_values(loaded_input_samples)

# Set the data from the imported file
output_samples.set_values(loaded_output_samples)


# Create integration set here.
integration_sample_set = samp.sample_set(dim_input)
integration_sample_set.set_domain(dim_range)
integration_sample_set = bsam.random_sample_set('random', 
        integration_sample_set, num_samples = 1E5)
integration_sample_set.estimate_volume_mc()

def randsamples(N):
    total_samples = len(loaded_input_samples)
    flag = 0
    while flag == 0:
        samples_selected = np.unique(np.random.randint(total_samples, size=N))
        if len(samples_selected) == N: # if you truly generated N unique samples
            flag = 1
    return samples_selected

# for each N value: 
# for N in N_values:
    # samples_selected = randsamples(N)
    # input_samples.set_values(loaded_input_samples[samples_selected])
    # output_samples.set_values(loaded_output_samples[samples_selected])

# for each reference (true) parameter, we solve the IP 
for kappa_0, kappa_1 in kappa_ref_locs:
    # compute state variable
    T = heatROD(1, amp, px, width, degree, T_R, kappa_0, kappa_1, 
                    rho, cap, nx, mesh, dt, t_stop, saveopt=False)
    Qref = np.array([T(xi) for xi in temp_locs]) # reference QoI value
    
    for QoI_indices in QoI_choice_list:
        # compute ref solution using all 10,000 samples. first we create our discretization object
        ref_inputs = input_samples.copy()
        ref_inputs.estimate_volume_mc()
        ref_outputs = output_samples.copy()
        ref_outputs._dim = len(QoI_indices)
        ref_outputs.set_values(ref_outputs._values[:, QoI_indices])
        
        ref_discretization = samp.discretization(ref_inputs, ref_outputs)
        # now we define our density on the output space
        simpleFunP.regular_partition_uniform_distribution_rectangle_scaled(ref_discretization, 
                Q_ref =  Qref[QoI_indices], rect_scale = rect_scale )
        # and finally, we calculate the probabilities on the input space
        calculateP.prob(ref_discretization)
        # (optional) plotting code
        # thing_to_plot = ref_discretization._input_sample_set
        # (bins, marginals2D) = plotP.calculate_2D_marginal_probs(thing_to_plot, nbins = [40, 40])    
        # plotP.plot_2D_marginal_probs(marginals2D, bins, thing_to_plot, 
        #                             filename = "q%drefsol_kappa[%.2f,%.2f]"%(QoI_indices[1], kappa_0, kappa_1),
        #                             lam_ref=np.array([kappa_0, kappa_1]), file_extension = ".eps", plot_surface=False)
        
        
# go through num_samples_list and compute est solution for each of these,


# compute HD between ref and est solution 

