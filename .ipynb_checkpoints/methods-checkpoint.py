#!/usr/bin/env python3

from GPyOpt.optimization.optimizer import OptLbfgs
from GPyOpt.core.task.space import Design_space
import itertools

def indicator_boundaries(bounds, d):

    """
    Check if all the values of d are in the domain.

    bounds: GPyOpt bounds
    d: proposed design

    """

    bounds = np.array(bounds)
    check = bounds.T - d
    low = all(i <= 0 for i in check[0])
    high = all(i >= 0 for i in check[1])

    if low and high:
        ind = 1.
    else:
        ind = 0.

    return np.array([[ind]])

def fun_dfun(obj, space, d):

    """
    Return posterior predictive + posterior predictive gradients.

    obj: GPyOpt object
    space: GPyOpt space
    d: proposed design

    """

    mask = space.indicator_constraints(d)

    pred = obj.model.predict_withGradients(d)[0][0][0]
    d_pred = obj.model.predict_withGradients(d)[2][0]

    return float(pred*mask), d_pred*mask

def get_GP_optimum(obj):

    """
    Obtain optimum from GPyOpt object.

    obj: GPyOpt object
    """

    # Define space
    space = Design_space(obj.domain, obj.constraints)
    bounds = space.get_bounds()

    # Get function to optimize + gradients
    # Also mask by everything that is allowed by the constraints
    fun = lambda d: fun_dfun(obj, space, d)[0]
    f_df = lambda d: fun_dfun(obj, space, d)

    # Specify Optimizer --- L-BFGS
    optimizer = OptLbfgs(space.get_bounds(), maxiter=1000)

    # Do the optimisation
    x, _ = optimizer.optimize(x0=obj.x_opt, f=fun, f_df=f_df)
    #x, _ = optimizer.optimize(x0=np.array([17,141,143]), f=fun, f_df=f_df)
    # TODO: MULTIPLE RE-STARTS FROM PREVIOUS BEST POINTS

    # Round values if space is discrete
    xtest = space.round_optimum(x)[0]

    if space.indicator_constraints(xtest):
        opt = xtest
    else:
        # Rounding mixed things up, so need to look at neighbours

        # Compute neighbours to optimum
        idx_comb = np.array(list(itertools.product([-1,0,1], repeat=len(bounds))))
        opt_combs = idx_comb + xtest

        # Evaluate
        GP_evals = list()
        combs = list()
        for idx, d in enumerate(opt_combs):

            cons_check = space.indicator_constraints(d)[0][0]
            bounds_check = indicator_boundaries(bounds, d)[0][0]

            if cons_check*bounds_check == 1:
                pred = obj.model.predict(d)[0][0][0]
                GP_evals.append(pred)
                combs.append(d)
            else:
                pass

        idx_opt = np.where(GP_evals == np.min(GP_evals))[0][0]
        opt = combs[idx_opt]

    return opt
