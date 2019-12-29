import bet.sample as samp
import bet.sensitivity.chooseQoIs as cqoi
import bet.sensitivity.gradients as grad
import bet.postProcess.plotP as plotP
import numpy as np
from estimatedist_setup import ref_input, ref_input_num, results_dir, BigN_values, dim_input
# mydisc = samp.load_discretization(
#     'results_heatrod_3/est_discretizations/rand_N_1280/rand_N_1280_trial_1')
# mydisc = grad.calculate_gradients_rbf(mydisc,1)
# cqoi.calculate_avg_skewness(mydisc._input_sample_set,[0,2])[0]

# qchoice = 1
for qchoice in range(1, 3):
    trial = 1
    n_samples = 2560
    BigN = BigN_values[0]**dim_input
    for M in [1]:
        file_name_ref = 'results_heatrod_3/ref_solutions/reg_BigN_40000/reg_M_%d/SolQoI_choice_%d-reg_M_%d_reg_BigN_%d' % (
            M, qchoice, M, BigN)
        file_name_est = 'results_heatrod_3/est_solutions/QoI_choice_%d/reg_M_%d/rand_N_%d/SolQoI_choice_%d-reg_M_%d_rand_N_%d_trial_%d' % (
            qchoice, M, n_samples, qchoice, M, n_samples, trial)

        # file_name_diff = 'results_heatrod_2/diff_t1_N20'
        # mysamp_diff = samp.load_sample_set(file_name_diff)

        mydisc_ref = samp.load_discretization(file_name_ref)
        mydisc_est = samp.load_discretization(file_name_est)
        mydisc_sk = grad.calculate_gradients_rbf(
            mydisc_ref, num_centers=20, num_neighbors=5)
        print(cqoi.calculate_avg_skewness(
            mydisc_sk._input_sample_set, [0, 1])[0])

        input_samples_ref = mydisc_ref._input_sample_set
        print(input_samples_ref.get_values()[0:5])
        input_samples_est = mydisc_est._input_sample_set
        print(input_samples_est.get_values()[0:5])
        # calculate 2d marginal probs, plot
        (bins, marginals2D) = plotP.calculate_2D_marginal_probs(
            input_samples_ref, nbins=[40, 40])
        plotP.plot_2D_marginal_probs(marginals2D, bins, input_samples_est, filename="%s/figures/M%d/refheat_pt%dQ%d_M%dN%d" % (results_dir, M, ref_input_num, qchoice, M, BigN),
                                     lam_ref=ref_input[0], file_extension=".png", plot_surface=False)

        (bins, marginals2D) = plotP.calculate_2D_marginal_probs(
            input_samples_est, nbins=[40, 40])
        plotP.plot_2D_marginal_probs(marginals2D, bins, input_samples_est, filename="%s/figures/M%d/estheat_pt%dQ%d_M%dN%d" % (results_dir, M, ref_input_num, qchoice, M, n_samples),
                                     lam_ref=ref_input[0], file_extension=".png", plot_surface=False)
    #
    # A_prob = set_A._probabilities[set_A_ptr]
    # A_vol = set_A._volumes[set_A_ptr]
    #
    # B_prob = set_B._probabilities[set_B_ptr]
    # B_vol = set_B._volumes[set_B_ptr]
    #
    # # prevents divide by zero. adheres to convention that if a cell has
    # # zero volume due to inadequate emulation, then it will get assigned zero probability as well
    # A_samples_lost = len(A_vol[A_vol == 0])
    # B_samples_lost = len(B_vol[B_vol == 0])
    # A_vol[A_vol == 0] = 1
    # B_vol[B_vol == 0] = 1
    #
    # den_A = np.divide( A_prob, A_vol)
    # den_B = np.divide( B_prob, B_vol )
    #
    # diff = (np.sqrt(den_A) - np.sqrt(den_B) )**2
    # C = integration_sample_set.copy()
    # C.set_probabilities(diff)
    # (bins, marginals2D) = plotP.calculate_2D_marginal_probs(C, nbins = [40, 40])
    # plotP.plot_2D_marginal_probs(marginals2D, bins, C, filename = "heattestmap_diff",
    #                              file_extension = ".png", plot_surface=False)

    # (bins, marginals2D) = plotP.calculate_2D_marginal_probs(input_samples_est, nbins = [40, 40])
    # plotP.plot_2D_marginal_probs(marginals2D, bins, input_samples_est, filename = "heat%dtestmap_diff"%qchoice,
    #                              lam_ref=ref_input[0], file_extension = ".png", plot_surface=False)

    # smooth 2d marginals probs (optional)
    # marginals2D = plotP.smooth_marginals_2D(marginals2D, bins, sigma=0.2)

