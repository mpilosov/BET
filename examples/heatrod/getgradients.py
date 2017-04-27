import scipy.io as sio
from extractqoivals import filename, temp_locs
import bet.sample as samp
import bet.sensitivity.gradients as bgrad
import bet.sensitivity.chooseQoIs as cqoi
import bet.sampling.basicSampling as bsam
from dolfin import *
import numpy as np
from simulation_2kappas_setup import *
from heatROD import *



num_centers_grad = 50
# create sample set for gradient computations
grad_centers_sample_set = samp.sample_set(2)
grad_centers_sample_set.set_domain( np.array([[0.01, 0.2],[0.01, 0.2]]) )
grad_centers_sample_set = bsam.random_sample_set('random', 
        grad_centers_sample_set, num_samples = num_centers_grad)
cluster_input_set = bgrad.pick_cfd_points(grad_centers_sample_set, 0.0025*np.ones(2))

# now that we've determined our clusters for the CFD, we need to run the heatrod
# and for each sample, compute the functional values (temp at a point) 
functional_values = np.zeros( (cluster_input_set.check_num(), len(temp_locs)) )
# loop over each sample and call heatrod
for i in range( cluster_input_set.check_num() ):
    # print 'Sample : ', i, '(kappa_0, kappa_1) = ', cluster_input_set._values[i,:]
    kappa_0, kappa_1 = cluster_input_set._values[i,:]
    
    T = heatROD(i, amp, px, width, degree, T_R, kappa_0, kappa_1, rho, cap, nx, mesh, dt, t_stop, saveopt=False)
    # now that heatROD has been run for this specific combination of kappas, we 
    # write the QoI values as the outputs in the discretization.
    functional_values[i] = np.array([T(xi) for xi in temp_locs]) # take temperature values at points of interest

cluster_output_set = samp.sample_set( len(temp_locs) )
cluster_output_set.set_values(functional_values)
cluster_discretization = samp.discretization(cluster_input_set, cluster_output_set) # combine into discretization
# cluster_output_set = samp.sample_set(2)
# cluster_discretization = samp.discretization(cluster_input_set, cluster_input_set) # combine into discretization

center_discretization = bgrad.calculate_gradients_cfd(cluster_discretization) # finally, compute the gradients
input_samples_centers = center_discretization.get_input_sample_set() # pull center values


index1 = 0
for index2 in range(1+index1,len(temp_locs)):
    (specific_skewness, _) = cqoi.calculate_avg_skewness(input_samples_centers,
            qoi_set=[index1, index2])
    print 'The average skewness of the QoI map defined by indices (%2d, %2d)'%(index1, index2) + \
        ' is ' + str(specific_skewness)



matfile = sio.loadmat(filename)

input_samples = samp.sample_set(2)
loaded_input_samples = matfile['samples']
input_samples.set_values(loaded_input_samples)

output_samples = samp.sample_set(len(temp_locs))
loaded_output_samples = matfile['data']
output_samples.set_values(loaded_output_samples)

samp_disc = samp.discretization(input_samples, output_samples)
center_discretization = bgrad.calculate_gradients_rbf(samp_disc, num_centers=100, num_neighbors = 5)
input_samples_centers = center_discretization.get_input_sample_set() # pull center values

index1 = 0
for index2 in range(1+index1,len(temp_locs)):
    (specific_skewness, _) = cqoi.calculate_avg_skewness(input_samples_centers,
            qoi_set=[index1, index2])
    print 'The average skewness of the QoI map defined by indices (%2d, %2d)'%(index1, index2) + \
        ' is ' + str(specific_skewness)


# # SAVE input_samples_centers
# centers_filename = 'input_sample_centers_grads'
# samp.save_sample_set(input_samples_centers,centers_filename)