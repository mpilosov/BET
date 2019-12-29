import os
import errno
import numpy as np
import bet.sample as samp
import bet.sampling.basicSampling as bsam
import bet.calculateP.simpleFunP as simpleFunP
import bet.calculateP.calculateP as calculateP
from heatrod import make_model

import bet.postProcess.compareP as compP

# temporary - for plotting integrand in HD
# import bet.postProcess.plotP as plotP


def ensure_path_exists(folder):
    try:
        os.makedirs(folder)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def mc_Hellinger(integration_sample_set, set_A, set_A_ptr, set_B, set_B_ptr):
    # Aset, Bset are sample_set type objects, to be evaluated at the points in
    # integration_sample_set. the pointers are from integration set into the others
    if isinstance(set_A, samp.discretization):
        set_A = set_A.get_input_sample_set()
    if isinstance(set_B, samp.discretization):
        set_B = set_B.get_input_sample_set()
    num_int_samples = integration_sample_set.check_num()

    A_prob = set_A._probabilities[set_A_ptr]
    A_vol = set_A._volumes[set_A_ptr]

    B_prob = set_B._probabilities[set_B_ptr]
    B_vol = set_B._volumes[set_B_ptr]

    # prevents divide by zero. adheres to convention that if a cell has
    # zero volume due to inadequate emulation, then it will get assigned zero probability as well
    A_samples_lost = len(A_vol[A_vol == 0])
    B_samples_lost = len(B_vol[B_vol == 0])
    A_vol[A_vol == 0] = 1
    B_vol[B_vol == 0] = 1

    den_A = np.divide(A_prob, A_vol)
    den_B = np.divide(B_prob, B_vol)

    # B_prob[A_prob == 0] = 0
    if B_samples_lost > 0:
        print('\t samples lost = %4d' % (B_samples_lost))
    if A_samples_lost > 0:
        print('\t !!!!! Integration samples lost = %4d' % (A_samples_lost))

    diff = (np.sqrt(den_A) - np.sqrt(den_B))**2
    C = integration_sample_set.copy()
    C.set_probabilities(diff)
    # (bins, marginals2D) = plotP.calculate_2D_marginal_probs(C, nbins = [40, 40])
    # plotP.plot_2D_marginal_probs(marginals2D, bins, C, filename = "heattestmap_diff",
    #                              file_extension = ".eps", plot_surface=False)

    return 0.5*(1./num_int_samples)*np.sum((np.sqrt(den_A) - np.sqrt(den_B))**2), C
    # THIS RETURNS THE SQUARE OF THE HELLINGER METRIC

    # return np.sqrt(0.5*(1./(num_int_samples - B_samples_lost))*np.sum( (np.sqrt(den_A) - np.sqrt(den_B) )**2 ) )


def invert_using(My_Discretization, Partition_Discretization, Emulated_Discretization, QoI_indices, Emulate=False):
    # Take full discretization objects, a set of indices for the QoI you want
    # to use to perform inversion, and then do so by redefining the output spaces
    # based on these indices and solving the problem as usual.
    # My_Discretization is copied, the other two are pass-by-reference

    # THIS IS FOR THE RECOVERY OF A DISTRIBUTION PROBLEM

    input_samples = My_Discretization._input_sample_set.copy()
    output_samples = My_Discretization._output_sample_set.copy()

    output_samples._dim = len(QoI_indices)
    output_samples.set_values(output_samples._values[:, QoI_indices])

    partition_output_samples = Partition_Discretization._output_sample_set.copy()
    partition_output_samples._dim = len(QoI_indices)
    partition_output_samples.set_values(
        partition_output_samples._values[:, QoI_indices])

    emulated_output_samples = Emulated_Discretization._output_sample_set.copy()
    emulated_output_samples._dim = len(QoI_indices)
    emulated_output_samples.set_values(
        emulated_output_samples._values[:, QoI_indices])

    output_samples.global_to_local()
    partition_output_samples.global_to_local()
    emulated_output_samples.global_to_local()
    partition_discretization = samp.discretization(input_sample_set=Partition_Discretization._input_sample_set,
                                                   output_sample_set=partition_output_samples)
    emulated_discretization = samp.discretization(input_sample_set=Emulated_Discretization._input_sample_set,
                                                  output_sample_set=emulated_output_samples)
    my_discretization = samp.discretization(input_sample_set=input_samples,
                                            output_sample_set=output_samples)
    if Emulate:
        my_discretization.set_emulated_input_sample_set(
            My_Discretization._emulated_input_sample_set.copy())

    # Compute the simple function approximation to the distribution on the data space of interest
    simpleFunP.user_partition_user_distribution(my_discretization,
                                                partition_discretization,
                                                emulated_discretization)

    if Emulate:
        # calculateP.prob_on_emulated_samples(my_discretization)
        calculateP.prob_with_emulated_volumes(my_discretization)
    else:
        calculateP.prob(my_discretization)

    return my_discretization


def invert_rect_using(My_Discretization, QoI_indices, Qref, rect, cells_per_dimension=1, Emulate=False):
    # Take full discretization objects, a set of indices for the QoI you want
    # to use to perform inversion, and then do so by redefining the output spaces
    # based on these indices and solving the problem as usual.
    # My_Discretization is copied, the other two are pass-by-reference

    # THIS IS FOR THE INVERTING A REFERENCE POINT PROBLEM

    input_samples = My_Discretization._input_sample_set.copy()
    output_samples = My_Discretization._output_sample_set.copy()

    output_samples._dim = len(QoI_indices)
    output_samples.set_values(output_samples._values[:, QoI_indices])

    output_samples.global_to_local()

    my_discretization = samp.discretization(input_sample_set=input_samples,
                                            output_sample_set=output_samples)
    if Emulate:
        my_discretization.set_emulated_input_sample_set(
            My_Discretization._emulated_input_sample_set.copy())

    # Compute the simple function approximation to the distribution on the data space of interest

    simpleFunP.regular_partition_uniform_distribution_rectangle_scaled(my_discretization,
                                                                       Q_ref=Qref[QoI_indices],
                                                                       rect_scale=rect,
                                                                       cells_per_dimension=cells_per_dimension)

    # simpleFunP.regular_partition_uniform_distribution_rectangle_size(my_discretization,
    #         Q_ref =  Qref[QoI_indices],
    #         rect_size = rect_size,
    #         cells_per_dimension = cells_per_dimension)

    if Emulate:
        # calculateP.prob_on_emulated_samples(my_discretization)
        calculateP.prob_with_emulated_volumes(my_discretization)
    else:
        calculateP.prob(my_discretization)

    return my_discretization
