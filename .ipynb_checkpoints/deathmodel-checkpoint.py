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

# Obtain Death model prior samples
mu, sigma = 1, 1
lower, upper = 0, 50
trunc = truncnorm((lower-mu)/sigma, (upper-mu)/sigma, loc=mu, scale=sigma)
prior_death = trunc.rvs(size=NS)

# Define the domain for BO
domain_death = [{'name': 'var_1', 'type': 'continuous', 'domain': (0.01, 4.00), 'dimensionality':int(DIMS)}]

# Define the constraints for BO
# Time cannot go backwards
if DIMS==1:
    constraints_death = None
elif DIMS>1:
    constraints_death = list()
    for i in range(1,DIMS):
        dic = {'name':'constr_{}'.format(i), 'constraint':'x[:,{}]-x[:,{}]'.format(i-1, i)}
        constraints_death.append(dic)
else:
    raise ValueError()

# ----- RUN MODEL ----- #

# Define the simulator model
truth_death = 1.5
model_death = simulator.DeathModelMultiple(truth_death, 50)

BED_death = staticdesign.StaticBED(prior_death, model_death, domain=domain_death, constraints=constraints_death, num_cores=num_cores)
BED_death.optimisation(init_num=INIT_NUM, max_iter=MAX_ITER)

# ---- SAVE MODEL ------ #
file = './deathmodel_dim{}'.format(DIMS)
BED_death.save(filename=file)
