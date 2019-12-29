from estimatedist_setup import *

# if you created (new) reference discretizations in the run, do this
if create_ref_disc or create_int_sets:
    print('Building Pointers from Integration Sets into Reference Meshes')
    # Generate io_pointer for integration sets
    for BigN in BigN_values:
        BigNval = BigN**(1 + (dim_input-1)*(reference_mesh_type == 'reg'))
        ref_sol_dir_2 = ref_sol_dir + \
            '%s_BigN_%d' % (reference_mesh_type, BigNval) + '/'
        print('\t Working on BigN = %d ... ' % (BigNval))
        for int_mesh_num in range(len(I_values)):
            Ival = I_values[int_mesh_num]**(1 + (dim_input-1)
                                            * (integration_mesh_type == 'reg'))
            filename = ref_sol_dir_2 + 'ptr_from_' + \
                '%s_I_%d' % (integration_mesh_type, Ival) + '_to_' + \
                '%s_BigN_%d' % (reference_mesh_type, BigNval)

            integration_sets_filename = cwd + '/' + results_dir + '/' + \
                sub_dirs[1] + '/' + \
                '%s_IntSet_%d' % (integration_mesh_type, Ival)
            int_mesh_set = samp.load_sample_set(integration_sets_filename)

            ref_filename = ref_sol_dir_2 + 'Reference_Disc-' + \
                '%s_BigN_%d' % (reference_mesh_type, BigNval)
            Ref_D = samp.load_discretization(ref_filename)

            (_, ptr) = Ref_D._input_sample_set.query(int_mesh_set._values)
            np.save(filename, ptr)
        # print ref_sol_int_ptrs_filenames
    print('Pointers from Integration Sets to Reference Meshes Created\n\n')

if create_est_discs or create_int_sets:
    print('Building Pointers from Integration Sets into Estimate Meshes')
    # Compute Pointers from Integration Sets into Estimates
    for N in N_values:
        Nval = N**(1 + (dim_input-1)*(estimate_mesh_type == 'reg'))
        est_disc_dir = cwd + '/' + results_dir + '/' + sub_dirs[2] + '/' + '%s_N_%d' % (
            estimate_mesh_type, Nval) + '/'  # all trial discretizations inside this folder
        print('\t Working on N = %d ... ' % (Nval))
        for trial in range(num_trials):
            est_disc_filename = est_disc_dir + \
                '%s_N_%d' % (estimate_mesh_type, Nval) + '_trial_%d' % (trial)
            My_D = samp.load_discretization(est_disc_filename)
            Input_Samples = My_D.get_input_sample_set()
            # Generate io_pointer for integration sets
            for int_mesh_num in range(len(I_values)):
                Ival = I_values[int_mesh_num]**(1 + (dim_input-1)
                                                * (integration_mesh_type == 'reg'))
                filename = est_disc_dir + 'ptr_from_' + '%s_I_%d' % (integration_mesh_type, Ival) + \
                    '_to_' + '%s_N_%d' % (estimate_mesh_type,
                                          Nval) + '_trial_%d' % (trial)

                integration_sets_filename = cwd + '/' + results_dir + '/' + \
                    sub_dirs[1] + '/' + \
                    '%s_IntSet_%d' % (integration_mesh_type, Ival)

                int_mesh_set = samp.load_sample_set(integration_sets_filename)
                (_, ptr) = Input_Samples.query(int_mesh_set._values)
                np.save(filename, ptr)
            print('\t \t Completed trial %d' % (trial))
        print('Pointers from Integration Sets to N = %d Estimates Created for all trials\n' % (Nval))
