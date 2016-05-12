# Copyright (C) 2016 The BET Development Team

# -*- coding: utf-8 -*-

import numpy as np
import os, sys
import scipy.io as sio
import bet.util as util
from bet.Comm import comm

# Parameter space is nD
# Data space is n/2 D

def main(io_file_name):
    # read in input from file
    io_mdat = sio.loadmat(io_file_name)
    input = io_mdat['input']
    # localize input
    input_local = np.array_split(input, comm.size)[comm.rank]
    # model is y = x[:, 0:dim/2 ] + x[:, dim/2:]
    output_local = sum(np.split(input_local, 2, 1))
    # save output to file
    io_mat['output'] = util.get_global_values(output_local)
    if comm.rank == 0:
        sio.savemat(io_file_name, io_mdat)

def usage():
    print "usage: [io_file]"

if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(sys.argv[1])
    else:
        usage()
