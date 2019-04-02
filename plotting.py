#!/usr/bin/env python3

import numpy as np

# Matplotlib parameters
import matplotlib.pyplot as plt
from matplotlib import rc
# Style
plt.style.use('default')
plt.rcParams['figure.figsize'] = (16.0, 8.0)
plt.rcParams.update({'font.size': 16})
# Fonts
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

# Utility information
from GPyOpt.acquisitions import AcquisitionEI
from GPyOpt.util.general import normalize

# ------- UTILITY PLOT ------- #
