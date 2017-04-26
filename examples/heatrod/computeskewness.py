from extractqoivals import temp_locs
import bet.sample as sample
import bet.sensitivity.gradients as bgrad
import bet.sensitivity.chooseQoIs as cqoi
from getgradients import centers_filename, temp_locs
# LOAD sample centers

input_samples_centers = sample.load_sample_set(centers_filename)
# Choose a specific set of QoIs to check the average skewness of
# for index1 in range(len(temp_locs)):
index1 = 0
for index2 in range(1+index1,len(temp_locs)):
    (specific_skewness, _) = cqoi.calculate_avg_skewness(input_samples_centers,
            qoi_set=[index1, index2])
    print 'The average skewness of the QoI map defined by indices (%2d, %2d)'%(index1, index2) + \
        ' is ' + str(specific_skewness)