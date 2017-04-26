import scipy.io as sio
from extractqoivals import filename, temp_locs
import bet.sample as sample


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

