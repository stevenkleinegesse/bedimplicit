#!/usr/bin/env python3

import numpy as np
from scipy.stats import truncnorm

import simulator
import staticdesign

# ---- GLOBAL PARAMS ------ #

# Set these to change performance of experimental design

# Number of CPU cores
num_cores = 1
# Set dimensions of design variable
DIMS = 1

# Number of prior samples (higher => more accurate posterior)
NS = 500
# Max number of utility evaluations in B.O. (per core)
MAX_ITER = 10

# number of initial data points for B.O.
if num_cores > 5:
    INIT_NUM = num_cores
else:
    INIT_NUM = 5

# ----- SPECIFY MODEL ----- #

# Obtain SIR model prior samples
param_0 = np.random.uniform(0, 0.5, NS).reshape(-1, 1)
param_1 = np.random.uniform(0, 0.5, NS).reshape(-1, 1)
prior_sir = np.hstack((param_0, param_1))

# Define the domain for BO
domain_sir = [{'name': 'var_1', 'type': 'continuous', 'domain': (0.01, 4.00), 'dimensionality':int(DIMS)}]

# Define the constraints for BO
# Time cannot go backwards
if DIMS==1:
    constraints_sir = None
elif DIMS>1:
    constraints_sir = list()
    for i in range(1,DIMS):
        dic = {'name':'constr_{}'.format(i), 'constraint':'x[:,{}]-x[:,{}]'.format(i-1, i)}
        constraints_sir.append(dic)
else:
    raise ValueError()

# ----- RUN MODEL ----- #

# Define the simulator model
truth_sir = np.array([0.15, 0.05])
if DIMS==1:
    model_sir = simulator.SIRModel(truth_sir, N=50)
else:
    model_sir = simulator.SIRModelMultiple(truth_sir, N=50)

BED_sir = staticdesign.StaticBED(prior_sir, model_sir, domain=domain_sir, constraints=constraints_sir, num_cores=num_cores)
BED_sir.optimisation(init_num=INIT_NUM, max_iter=MAX_ITER)

# ---- SAVE MODEL ------ #
file = './sirmodel_dim'.format(DIMS)
BED_sir.save(filename=file)
