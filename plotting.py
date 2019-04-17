#!/usr/bin/env python3

import numpy as np
from scipy.stats import gaussian_kde

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

def plot_acquisition(obj, d_opt, filename = None):

    # GPyOpt part: Interpolation
    bounds = obj.space.get_bounds()
    x_grid = np.linspace(bounds[0][0], bounds[0][1], 1000)
    x_grid = x_grid.reshape(len(x_grid),1)
    AEI = AcquisitionEI(obj.model, obj.space)
    acqu = AEI._compute_acq(x_grid)
    acqu_normalized = (-acqu - min(-acqu))/(max(-acqu - min(-acqu)))
    m, s = obj.model.predict(x_grid)

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)

    # Plot Model
    # Note that we have to flip the GPyOpt utility
    ax.plot(x_grid, -m, 'k-',lw=1,alpha = 0.6)
    ax.plot(x_grid, -(m+np.sqrt(s)), 'k-', alpha = 0.2)
    ax.plot(x_grid, -(m-np.sqrt(s)), 'k-', alpha=0.2)
    ax.fill_between(x_grid.reshape(-1),
                -(m+np.sqrt(s)).reshape(-1),
                -(m-np.sqrt(s)).reshape(-1),
                color='steelblue', alpha=0.5, label=r'68\% C.L.')

    # Plot Evals
    X = obj.X
    Y = normalize(obj.Y)
    ax.plot(X, -1 * Y, 'r.', markersize=10, label='Bayes. Opt. Evaluations')

    # Plot optimum
    ax.axvline(d_opt, ls='--', c='g', label='Optimal Design')

    ax.set_xlabel(r'Design variable d')
    ax.set_ylabel(r'$U(d)$')
    ax.legend(loc='bottom right',prop={'size': 14})
    ax.grid(True, ls='--')
    # ax.set_title('Utility Function')
    # ax.tick_params(labelsize=25)

    plt.tight_layout()

    if filename:
        plt.savefig('{}.pdf'.format(filename))
        plt.savefig('{}.png'.format(filename))
    else:
        plt.savefig('./utility.pdf')
        plt.savefig('./utility.png')

# ------- POSTERIOR PLOTS ------- #

def plot_posterior(prior_samples, ratios, model, truth, filename = None):

    # Get some posterior samples from ratios

    ww = np.array(ratios)
    pp = prior_samples

    ww[ww == np.inf] = 0
    ws_norm = ww / np.sum(ww)

    K = 10000
    post = list()
    for _ in range(K):
        cat = np.random.choice(range(len(ws_norm)), p=ws_norm)
        post.append(pp[cat])
    post = np.array(post)

    if model=='death':

        # define kde smoothing and grid
        smooth = 0.35
        xs = np.linspace(np.min(prior_samples),np.max(prior_samples),1000)

        # get kde on grid
        density = gaussian_kde(post)
        density.covariance_factor = lambda : smooth
        density._compute_covariance()
        kde = density(xs)

        # make plot
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)
        ax.plot(xs, kde, lw=2, label='Posterior KDE')
        ax.axvline(truth, ls='--', c='r', lw=2, label=r'$b_{\text{true}}$', alpha=0.5)
        ax.grid(True, ls='--')
        ax.tick_params(labelsize=20)
        ax.set_xlabel(r'b', size=20)
        ax.set_ylabel(r'Posterior Density', size=20)
        ax.legend(prop={'size': 17})

        plt.tight_layout()

    elif model=='sir':

        # define grid
        smooth = 0.35
        xmin = np.min(prior_samples[:,0])
        xmax = np.max(prior_samples[:,0])
        ymin = np.min(prior_samples[:,1])
        ymax = np.max(prior_samples[:,1])
        X, Y = np.mgrid[xmin:xmax:500j, ymin:ymax:500j]
        positions = np.vstack([X.ravel(), Y.ravel()])

        # get kde on grid
        values = np.vstack([post[:,0], post[:,1]])
        kernel = gaussian_kde(values)
        kernel.covariance_factor = lambda : smooth
        kernel._compute_covariance()
        Z = np.reshape(kernel(positions).T, X.shape)

        # make plot
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)
        ax.contour(Z.T, cmap=plt.cm.viridis, extent=[xmin, xmax, ymin, ymax])
        ax.grid(True, ls='--')
        ax.tick_params(labelsize=25)
        ax.set_xlabel(r'$\beta$', size=30)
        ax.set_ylabel(r'$\gamma$', size=30)
        ax.set_ylim([0,0.15])
        ax.scatter(truth[0], truth[1], c='r', marker=r'x', s=100)

        plt.tight_layout()

    else:
        raise NotImplementedError()

    if filename:
        plt.savefig('{}.pdf'.format(filename))
        plt.savefig('{}.png'.format(filename))
    else:
        plt.savefig('./{}model_posterior.pdf'.format(model))
        plt.savefig('./{}model_posterior.png'.format(model))
