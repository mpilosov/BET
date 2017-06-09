#! /usr/bin/env python

# Copyright (C) 2014-2016 The BET Development Team

#TODO Edit description 
"""
This example solves a stochastic inverse problem for a
linear 3-to-2 map. We refer to the map as the QoI map,
or just a QoI. We refer to the range of the QoI map as
the data space.
The 3-D input space is discretized with i.i.d. uniform
random samples or a regular grid of samples.
We refer to the input space as the
parameter space, and use parameter to refer to a particular
point (e.g., a particular random sample) in this space.
A reference parameter is used to define a reference QoI datum
and a uniform probability measure is defined on a small box
centered at this datum.
The measure on the data space is discretized either randomly
or deterministically, and this discretized measure is then
inverted by BET to determine a probability measure on the
parameter space whose support contains the measurable sets
of probable parameters.
We use emulation to estimate the measures of sets defined by
the random discretizations.
1D and 2D marginals are calculated, smoothed, and plotted.
"""

import numpy as np
import bet.calculateP.simpleFunP as simpleFunP
import bet.calculateP.calculateP as calculateP
import bet.postProcess.plotP as plotP
import bet.postProcess.plotDomains as plotD
import bet.sample as samp
import bet.sampling.basicSampling as bsam
from myModel import my_model

# Define the sampler that will be used to create the discretization
# object, which is the fundamental object used by BET to compute
# solutions to the stochastic inverse problem.
# The sampler and my_model is the interface of BET to the model,
# and it allows BET to create input/output samples of the model.
sampler = bsam.sampler(my_model)

# Initialize 4-dimensional input parameter sample set object
input_samples = samp.sample_set(4)

# Set parameter domain (ordering is [a, b, c, n] )
input_samples.set_domain(np.array([ [-10.0, 10.0],
                                    [-10.0, 10.0], 
                                    [-5.0, 5.0], 
                                    [-3.0, 3.0] ]))

'''
Suggested changes for user:

Try with and without random sampling.

If using random sampling, try num_samples = 1E3 and 1E4.
What happens when num_samples = 1E2?
Try using 'lhs' instead of 'random' in the random_sample_set.

If using regular sampling, try different numbers of samples
per dimension.
'''
# Generate samples on the parameter space
num_input_samples = 18E4 # Note 20,000 samples is similar to what the regular grid would use
grid_factor = 1 # for regular sampling only. how many per interval.
randomSampling = True
if randomSampling is True:
    input_samples = sampler.random_sample_set('random', input_samples, num_samples=num_input_samples)
else:
    # if regular sampling, use the integer grid in uncertainty box. 24,000 samples.
    input_samples = sampler.regular_sample_set(input_samples, num_samples_per_dim=grid_factor*np.array([20, 20, 10, 6])) 

'''
Suggested changes for user:

A standard Monte Carlo (MC) assumption is that every Voronoi cell
has the same volume. If a regular grid of samples was used, then
the standard MC assumption is true.

See what happens if the MC assumption is not assumed to be true, and
if different numbers of points are used to estimate the volumes of
the Voronoi cells.
'''
MC_assumption = True
start_time = time.time()
# Estimate volumes of Voronoi cells associated with the parameter samples
if MC_assumption is False:
    input_samples.estimate_volume(n_mc_points=num_input_samples*50.0)
else:
    input_samples.estimate_volume_mc()

# Create the discretization object using the input samples
my_discretization = sampler.compute_QoI_and_create_discretization(input_samples,
                                               savefile = '4to4_discretization.txt.gz')
'''
Suggested changes for user:

Try different reference parameters.
'''
# Define the reference parameter
a = 20*np.random.rand()-10 # choose any a between -10 and 10
# pick any one of the analytical solutions [a,b,c,n]
ref_param = np.array([[a, 1.0, -a, 0.0]])
# ref_param = np.array([[a, 1.0-a, 0.0, 1.0]])
Q_ref =  my_model(np.array(ref_param))
print ref_param[0], '=>', Q_ref[0]

'''
Suggested changes for user:

Try different ways of discretizing the probability measure on D defined as a uniform
probability measure on a rectangle (since D is 2-dimensional) centered at Q_ref whose
size is determined by scaling the circumscribing box of D.
'''
randomDataDiscretization = False
if randomDataDiscretization is False:
    simpleFunP.regular_partition_uniform_distribution_rectangle_scaled(
        data_set=my_discretization, Q_ref=Q_ref, rect_scale=0.25,
        cells_per_dimension = 3)
else:
    simpleFunP.uniform_partition_uniform_distribution_rectangle_scaled(
        data_set=my_discretization, Q_ref=Q_ref, rect_scale=0.25,
        M=50, num_d_emulate=1E5)

# calculate probablities
calculateP.prob(my_discretization)

########################################
# Post-process the results
########################################
'''
Suggested changes for user:

At this point, the only thing that should change in the plotP.* inputs
should be either the nbins values or sigma (which influences the kernel
density estimation with smaller values implying a density estimate that
looks more like a histogram and larger values smoothing out the values
more).

There are ways to determine "optimal" smoothing parameters (e.g., see CV, GCV,
and other similar methods), but we have not incorporated these into the code
as lower-dimensional marginal plots generally have limited value in understanding
the structure of a high dimensional non-parametric probability measure.
'''
plot_disc = 10 # plotting discretization

# calculate 2d marginal probs
(bins2D, marginals2D) = plotP.calculate_2D_marginal_probs(input_samples,
                                                        nbins = plot_disc*np.ones(4))

# smooth 2d marginals probs (optional)
# marginals2D = plotP.smooth_marginals_2D(marginals2D, bins, sigma=0.2)

# plot 2d marginals probs
plotP.plot_2D_marginal_probs(marginals2D, bins, input_samples, filename = "linearMap",
                             lam_ref=ref_param[0], file_extension = ".eps", plot_surface=False)

# calculate 1d marginal probs
(bins1D, marginals1D) = plotP.calculate_1D_marginal_probs(input_samples,
#                                                         nbins = plot_disc*np.ones(4))
                                                        nbins = plot_disc*np.array([20, 20, 10, 6]))
# smooth 1d marginal probs (optional)
marginals1D = plotP.smooth_marginals_1D(marginals1D, bins, sigma=0.1)

# plot 2d marginal probs
plotP.plot_1D_marginal_probs(marginals1D, bins, input_samples, filename = "linearMap",
                             lam_ref=ref_param[0], file_extension = ".eps")
