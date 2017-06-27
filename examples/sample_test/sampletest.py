import bet.sample as samp
import bet.sampling.basicSampling as bsam
import numpy as np

def my_model(samples):
    return samples[:,0]

sampler = bsam.sampler(my_model)

input_samples = samp.sample_set(2)
input_samples.set_domain(np.array([[0,1],[0,1]]))

num_input_samples = int(1E4)
input_samples = sampler.random_sample_set('random', input_samples, num_samples = num_input_samples)
#input_samples.estimate_volume(n_mc_points=num_input_samples*50)
input_samples.estimate_volume_mc()
my_disc = sampler.compute_QoI_and_create_discretization(input_samples)

