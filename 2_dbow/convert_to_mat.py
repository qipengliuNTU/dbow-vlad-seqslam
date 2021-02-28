#!/usr/bin/env python3

from scipy.io import savemat
import numpy as np
import sys

input_path = sys.argv[1]
output_path = sys.argv[2]

bow_data = np.loadtxt(input_path)
bow_data = 1 - bow_data
savemat(output_path, {'D': bow_data})
