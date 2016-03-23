# Copyright (C) 2014-2015 The BET Development Team

"""
This example generates uniform random samples in the unit hypercube and
corresponding QoIs (data) generated by a linear map Q.  We then calculate the
gradients using an RBF scheme and use the gradient information to choose the
optimal set of 2 (3, 4, ... input_dim) QoIs to use in the inverse problem.

Every real world problem requires special attention regarding how we choose
*optimal QoIs*.  This set of examples (examples/sensitivity/linear) covers
some of the more common scenarios using easy to understand linear maps.

In this *condnum_binratio* example we choose *optimal QoIs* to be the set of QoIs
of size input_dim that has optimal skewness properties which will yield an
inverse solution that can be approximated well.  The uncertainty in our data is
relative to the range of data measured in each QoI (bin_ratio).
"""

import numpy as np
import bet.sensitivity.gradients as grad
import bet.sensitivity.chooseQoIs as cQoI
import bet.calculateP.simpleFunP as simpleFunP
import bet.calculateP.calculateP as calculateP
import bet.postProcess.postTools as postTools
import bet.Comm as comm
import bet.sample as sample

# Let Lambda be a 5 dimensional hypercube
input_dim = 5
output_dim = 10
num_samples = 1E5
num_centers = 10

# Let the map Q be a random matrix of size (output_dim, input_dim)
np.random.seed(0)
Q = np.random.random([output_dim, input_dim])

# Choose random samples in parameter space to solve the model
input_set = sample.sample_set(input_dim)
input_set_centers = sample.sample_set(input_dim)
output_set = sample.sample_set(output_dim)

input_set._values = np.random.random([num_samples, input_dim])
input_set_centers._values = input_set._values[:num_centers]
output_set._values = Q.dot(input_set._values.transpose()).transpose()

# Calculate the gradient vectors at some subset of the samples.  Here the
# *normalize* argument is set to *True* because we are using bin_ratio to
# determine the uncertainty in our data.
input_set._jacobians = grad.calculate_gradients_rbf(input_set, output_set,
    input_set_centers, normalize=True)

# With these gradient vectors, we are now ready to choose an optimal set of
# QoIs to use in the inverse problem, based on optimal skewness properites of
# QoI vectors.  The most robust method for this is
# :meth:~bet.sensitivity.chooseQoIs.chooseOptQoIs_large which returns the
# best set of 2, 3, 4 ... until input_dim.  This method returns a list of
# matrices.  Each matrix has 10 rows, the first column representing the
# average condition number of the Jacobian of Q, and the rest of the columns
# the corresponding QoI indices.
best_sets = cQoI.chooseOptQoIs_large(input_set, volume=False)

###############################################################################

# At this point we have determined the optimal set of QoIs to use in the inverse
# problem.  Now we compare the support of the inverse solution using
# different sets of these QoIs.  We set Q_ref to correspond to the center of
# the parameter space.  We choose the set of QoIs to consider.

QoI_indices = [3, 4] # choose up to input_dim
#QoI_indices = [3, 6]
#QoI_indices = [0, 3]
#QoI_indices = [3, 5, 6, 8, 9]
#QoI_indices = [0, 3, 5, 8, 9]
#QoI_indices = [3, 4, 5, 8, 9]
#QoI_indices = [2, 3, 5, 6, 9]

# Restrict the data to have just QoI_indices
output_set._values = output_set._values[:, QoI_indices]
Q_ref = Q[QoI_indices, :].dot(0.5 * np.ones(input_dim))
# bin_ratio defines the uncertainty in our data
bin_ratio = 0.25

# Find the simple function approximation
(d_distr_prob, d_distr_samples, d_Tree) = simpleFunP.uniform_hyperrectangle(\
    data=output_set._values, Q_ref=Q_ref, bin_ratio=bin_ratio, center_pts_per_edge = 1)

# Calculate probablities making the Monte Carlo assumption
(P,  lam_vol, io_ptr) = calculateP.prob(samples=input_set._values,
     data=output_set._values, rho_D_M=d_distr_prob,
     d_distr_samples=d_distr_samples)

percentile = 1.0
# Sort samples by highest probability density and find how many samples lie in
# the support of the inverse solution.  With the Monte Carlo assumption, this
# also tells us the approximate volume of this support.
(num_samples, P_high, samples_high, lam_vol_high, data_high, sort) =\
    postTools.sample_highest_prob(top_percentile=percentile, P_samples=P,
    samples=input_set._values, lam_vol=lam_vol,data=output_set._values,sort=True)

# Print the number of samples that make up the highest percentile percent
# samples and ratio of the volume of the parameter domain they take up
if comm.rank == 0:
    print (num_samples, np.sum(lam_vol_high))
