import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.ticker import MultipleLocator
from sklearn.linear_model import Ridge
import seaborn as sns
sns.set_style('ticks')
sns.set_context('talk')
colors = sns.color_palette('Set1')


def sine(x):
    return np.sin(2*np.pi*x)


def rbf(x, mu, sigma=0.05):
    return np.exp(-(x - mu)**2/(2*sigma**2))


def plot_sine(noise_var=0.25):
    plt.close('all')
    plt.figure(figsize=(12, 6))

    major_locator = MultipleLocator(0.5)

    x = np.linspace(0, 1, 1000)
    y_mean = sine(x)

    x_samples = np.linspace(0, 1, 25)
    y_samples = sine(x_samples) + np.random.randn(25) * np.sqrt(.09)

    plt.subplot(1, 2, 1)
    plt.plot(x, y_mean, color='k')
    plt.plot(x_samples, y_samples, linestyle='none', marker='o', color=colors[0])
    sns.despine(offset=5, trim=True)
    plt.xlim(0, 1)
    plt.ylim(-2, 2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.gca().xaxis.set_major_locator(major_locator)
    plt.gca().yaxis.set_major_locator(major_locator)

    y = np.linspace(-2, 2, 1000)
    xx, yy = np.meshgrid(x, y)
    c = np.zeros_like(xx)

    for i, mu in enumerate(y_mean):
        c[:, i] = sp.stats.norm.pdf(yy[:, i], mu, noise_var)
    plt.subplot(1, 2, 2)
    plt.pcolormesh(xx, yy, c, cmap='Reds')
    sns.despine(offset=5, trim=True)
    plt.xlim(0, 1)
    plt.ylim(-2, 2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.gca().xaxis.set_major_locator(major_locator)
    plt.gca().yaxis.set_major_locator(major_locator)

    plt.tight_layout()

    plt.savefig(
        '/Users/lukewoloszyn/Personal/projects/newsite/content/images/1-function.png',
        transparent=True)


def plot_rbfs():
    plt.close('all')
    plt.figure(figsize=(12, 6))

    x = np.linspace(0, 1, 1000)
    example_x = 0.25
    mus = np.linspace(0, 1, 24)
    phi = []
    for mu in mus:
        plt.plot(x, rbf(x, mu), alpha=0.5)
        phi.append(rbf(example_x, mu))
    plt.plot(mus, phi, color='k', marker='o')
    plt.ylim(-0.01, 1.01)
    plt.xlabel('x')
    plt.ylabel('y')
    sns.despine(offset=5, trim=True)
    plt.tight_layout()
    plt.savefig('/Users/lukewoloszyn/Personal/projects/newsite/content/images/1-rbfs.png',
                transparent=True)


def plot_bias_variance(n_rbf=24,
                       n_per_alpha=1000,
                       n_points_train=25,
                       n_points_plot=100,
                       n_fits_plot=20,
                       n_alphas=3,
                       plot_type='ind_fits'):

    plt.close('all')
    if plot_type == 'ind_fits':
        fig = plt.figure(figsize=(12, 10))
    else:
        fig = plt.figure(figsize=(12, 6))

    # x that we will use to fit data
    x_tofit = np.linspace(0, 1, n_points_train)[:, np.newaxis]
    X_tofit = np.tile(x_tofit, (1, n_rbf))
    for j, mu in enumerate(np.linspace(0, 1, n_rbf)):
        X_tofit[:, j] = rbf(X_tofit[:, j], mu=mu)

    # x that we will use to plot data (higher res)
    x_toplot = np.linspace(0, 1, n_points_plot)[:, np.newaxis]
    X_toplot = np.tile(x_toplot, (1, n_rbf))
    x_toplot = x_toplot.ravel()
    X_toplot[:, 0] = np.ones(n_points_plot)
    for i, mu in enumerate(np.linspace(0, 1, n_rbf)):
        X_toplot[:, i] = rbf(X_toplot[:, i], mu=mu)

    if plot_type == 'ind_fits':
        alphas = np.logspace(-1.0, 1.0, n_alphas)
    elif plot_type == 'bv':
        alphas = np.logspace(-2.5, 1.0, n_alphas)
    bias_sq = np.zeros(n_alphas)
    variance = np.zeros(n_alphas)

    colors = sns.color_palette('Set1', 6)
    for actr, alpha in enumerate(alphas):
        Y = np.zeros((n_per_alpha, n_points_plot))
        for i in xrange(n_per_alpha):
            model = Ridge(alpha=alpha)

            y = sine(x_tofit.ravel())
            y += np.random.randn(n_points_train)*np.sqrt(.09)
            model.fit(X_tofit, y)
            y_pred = model.predict(X_toplot)
            Y[i, :] = y_pred

        # plotting
        if plot_type == 'ind_fits':
            ax1 = fig.add_subplot(n_alphas, 3, 3*actr+1)
            line_collection = LineCollection(
                [zip(x_toplot, y_toplot)
                 for y_toplot in Y[np.random.permutation(n_per_alpha)
                                   < n_fits_plot]],
                color=colors[0], linewidth=0.5, alpha=0.5
            )
            ax1.add_collection(line_collection)
            ax1.text(0.65, 0.85, '$\lambda$ = {:3.1f}'.format(alpha),
                     transform=ax1.transAxes)

            ax2 = fig.add_subplot(n_alphas, 3, 3*actr+2)
            ax2.fill_between(x_toplot.ravel(),
                             np.mean(Y, axis=0) - np.var(Y, axis=0),
                             np.mean(Y, axis=0) + np.var(Y, axis=0),
                             color=colors[0], alpha=0.5)

            ax3 = fig.add_subplot(n_alphas, 3, 3*actr+3)
            ax3.add_line(plt.Line2D(x_toplot, sine(x_toplot),
                                    color=colors[1], linewidth=1))
            ax3.add_line(plt.Line2D(x_toplot, np.mean(Y, axis=0),
                                    color=colors[0], linewidth=1))
            if actr == 0:
                ax3.text(0.4, 0.85, 'Average prediction',
                         transform=ax3.transAxes, color=colors[0])
                ax3.text(0.4, 0.75, 'True function',
                         transform=ax3.transAxes, color=colors[1])

            for ax in [ax1, ax2, ax3]:
                ax.set_xlim(0, 1)
                ax.set_ylim(-1.5, 1.5)
                sns.despine(ax=ax, offset=5, trim=True)
                major_locator = MultipleLocator(0.5)
                ax.xaxis.set_major_locator(major_locator)
                ax.yaxis.set_major_locator(major_locator)

            if actr == 2:
                ax1.set_xlabel('x')
                ax1.set_ylabel('y')

        bias_sq[actr] = np.mean(np.power(np.mean(Y, axis=0) - sine(x_toplot), 2))
        variance[actr] = np.mean(np.var(Y, axis=0))

    if plot_type == 'bv':
        plt.plot(np.log10(alphas), bias_sq, color=colors[0])
        plt.plot(np.log10(alphas), variance, color=colors[1])
        plt.plot(np.log10(alphas), variance+bias_sq, color=colors[2])
        plt.legend(['$(bias)^2$', '$variance$', '$(bias)^2 + variance$'], loc=9)
        plt.xlabel('$log_{10}(\lambda)$')
        plt.ylim(0, 0.15)
        sns.despine(offset=5, trim=True)

    plt.tight_layout()

    if plot_type == 'ind_fits':
        plt.savefig('/Users/lukewoloszyn/Personal/projects/newsite/content/images/1-ind_fits.png',
                    transparent=True)
    elif plot_type == 'bv':
        plt.savefig('/Users/lukewoloszyn/Personal/projects/newsite/content/images/1-bv.png',
                    transparent=True)
