import scipy.io as sio
from extractqoivals import filename, temp_locs
import bet.sample as samp

QoI_choice_list = [ [0, n+1] for n in range(len(skew_range))]

matfile = sio.loadmat(filename)
loaded_input_samples = matfile['samples']
loaded_output_samples = matfile['data']

# Initialize the necessary sample objects
input_samples = sample.sample_set(2)
output_samples = sample.sample_set( len(temp_locs) )

# Set the input sample values from the imported file
input_samples.set_values(loaded_input_samples)

# Set the data from the imported file
output_samples.set_values(loaded_output_samples)


# compute ref solution using all 10,000 samples.


# go through num_samples_list and compute est solution for each of these,


# compute HD between ref and est solution 

