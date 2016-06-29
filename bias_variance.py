import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.ticker import MultipleLocator
import seaborn as sns
from numpy.linalg import pinv
sns.set_style('ticks')
sns.set_context('talk')
COLORS = sns.color_palette('Set1', 6)


def sine(x):
    return np.sin(2*np.pi*x)


def rbf(x, mu, sigma=0.075):
    return np.exp(-(x - mu)**2/(2*sigma**2))


def plot_sine(noise_var=0.09):
    plt.close('all')
    fig = plt.figure(figsize=(12, 6))

    x = np.linspace(0, 1, 1000)
    y_mean = sine(x)

    x_samples = np.linspace(0, 1, 25)
    y_samples = sine(x_samples) + np.random.randn(25) * np.sqrt(noise_var)

    plt.subplot(1, 2, 1)
    plt.plot(x, y_mean, color='k')
    plt.plot(x_samples, y_samples, linestyle='none', marker='o', color=COLORS[0])

    y = np.linspace(-2, 2, 1000)
    xx, yy = np.meshgrid(x, y)
    c = np.zeros_like(xx)

    for i, mu in enumerate(y_mean):
        c[:, i] = sp.stats.norm.pdf(yy[:, i], mu, np.sqrt(noise_var))
    plt.subplot(1, 2, 2)
    plt.pcolormesh(xx, yy, c, cmap='Reds')

    major_locator = MultipleLocator(0.5)
    for ax in fig.get_axes():
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-2.01, 2.01)
        sns.despine(ax=ax, offset=5, trim=True)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.xaxis.set_major_locator(major_locator)
        ax.yaxis.set_major_locator(major_locator)

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
    plt.xlim(-0.01, 1.02)
    plt.ylim(-0.05, 1.05)
    plt.xlabel('x')
    plt.ylabel('y')
    sns.despine(offset=5, trim=True)
    plt.tight_layout()
    plt.savefig('/Users/lukewoloszyn/Personal/projects/newsite/content/images/1-rbfs.png',
                transparent=True)


def bias_variance(n_rbf=24,
                  n_per_alpha=1000,
                  n_points_train=25,
                  n_points_plot=100,
                  n_fits_plot=20,
                  n_alphas=3,
                  plot_type=None):

    plt.close('all')
    if plot_type == 'ind_fits':
        fig = plt.figure(figsize=(12, 10))
    elif plot_type == 'bv':
        fig = plt.figure(figsize=(12, 6))

    # x that we will use to plot data (higher res)
    x_toplot = np.linspace(0, 1, n_points_plot)[:, np.newaxis]
    X_toplot = np.tile(x_toplot, (1, n_rbf))
    x_toplot = x_toplot.ravel()
    for i, mu in enumerate(np.linspace(0, 1, n_rbf)):
        X_toplot[:, i] = rbf(X_toplot[:, i], mu=mu)

    alphas = np.logspace(-1.0, 1.0, n_alphas)
    bias_sq = np.zeros(n_alphas)
    variance = np.zeros(n_alphas)

    for actr, alpha in enumerate(alphas):
        model = CustomRidge(alpha=alpha)
        Y = np.zeros((n_per_alpha, n_points_plot))
        for i in xrange(n_per_alpha):
            # the x that we will use to fit data
            x_tofit = np.random.rand(n_points_train)[:, np.newaxis]
            X_tofit = np.tile(x_tofit, (1, n_rbf))
            x_tofit = x_tofit.ravel()
            for j, mu in enumerate(np.linspace(0, 1, n_rbf)):
                X_tofit[:, j] = rbf(X_tofit[:, j], mu=mu)

            y = sine(x_tofit) + np.random.randn(n_points_train)*np.sqrt(.09)
            model.fit(X_tofit, y)
            Y[i, :] = model.predict(X_toplot)

        if plot_type == 'ind_fits':
            ax = fig.add_subplot(n_alphas, 3, 3*actr+1)
            plot_ind_fits(ax, x_toplot, Y, alpha, n_fits_plot)
            ax = fig.add_subplot(n_alphas, 3, 3*actr+2)
            plot_fit_var(ax, x_toplot, Y)
            ax = fig.add_subplot(n_alphas, 3, 3*actr+3)
            plot_avg_pred(ax, x_toplot, Y, actr == 0)

        bias_sq[actr] = np.mean(np.power(np.mean(Y, axis=0) - sine(x_toplot), 2))
        variance[actr] = np.mean(np.var(Y, axis=0))

    if plot_type == 'ind_fits':
        major_locator = MultipleLocator(0.5)
        for i, ax in enumerate(fig.get_axes()):
            ax.set_xlim(0, 1)
            ax.set_ylim(-1.5, 1.5)
            sns.despine(ax=ax, offset=5, trim=True)
            ax.xaxis.set_major_locator(major_locator)
            ax.yaxis.set_major_locator(major_locator)
            if actr == 6:
                ax.set_xlabel('x')
                ax.set_ylabel('y')
    elif plot_type == 'bv':
        plot_bias_var(alphas, bias_sq, variance)

    if plot_type:
        plt.tight_layout()

    if plot_type == 'ind_fits':
        plt.savefig('/Users/lukewoloszyn/Personal/projects/newsite/content/images/1-ind_fits.png',
                    transparent=True)
    elif plot_type == 'bv':
        plt.savefig('/Users/lukewoloszyn/Personal/projects/newsite/content/images/1-bv.png',
                    transparent=True)


def plot_ind_fits(ax, x, Y, alpha, n_plot):
    line_collection = LineCollection(
        [zip(x, y) for y in Y[np.random.permutation(Y.shape[0]) < n_plot]],
        color=COLORS[0], linewidth=0.5, alpha=0.5
    )
    ax.add_collection(line_collection)
    ax.text(0.65, 0.85, '$\lambda$ = {:3.1f}'.format(alpha),
            transform=ax.transAxes)


def plot_fit_var(ax, x, Y):
    ax.fill_between(x.ravel(),
                    np.mean(Y, axis=0) - np.var(Y, axis=0),
                    np.mean(Y, axis=0) + np.var(Y, axis=0),
                    color=COLORS[0], alpha=0.5)


def plot_avg_pred(ax, x, Y, label=False):
    ax.add_line(plt.Line2D(x, sine(x),
                           color=COLORS[1], linewidth=1))
    ax.add_line(plt.Line2D(x, np.mean(Y, axis=0),
                           color=COLORS[0], linewidth=1))
    if label:
        ax.text(0.4, 0.85, 'Average prediction',
                transform=ax.transAxes, color=COLORS[0])
        ax.text(0.4, 0.75, 'True function',
                transform=ax.transAxes, color=COLORS[1])


def plot_bias_var(alphas, bias_sq, variance):
    plt.plot(np.log10(alphas), bias_sq, color=COLORS[0])
    plt.plot(np.log10(alphas), variance, color=COLORS[1])
    plt.plot(np.log10(alphas), variance+bias_sq, color=COLORS[2])
    #plt.plot(np.log(alphas), bias_sq, color=COLORS[0])
    #plt.plot(np.log(alphas), variance, color=COLORS[1])
    #plt.plot(np.log(alphas), variance+bias_sq, color=COLORS[2])
    plt.legend(['$(bias)^2$', '$variance$', '$(bias)^2 + variance$'], loc=9)
    plt.xlabel('$log_{10}(\lambda)$')
    plt.ylim(0, 0.15)
    sns.despine(offset=5, trim=True)


class CustomRidge(object):

    def __init__(self, alpha=1, fit_intercept=True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        # don't penalize intercept
        lambda_matrix = np.eye(X.shape[1]) * self.alpha
        if self.fit_intercept:
            lambda_matrix[0, 0] = 0
        self.coefs = pinv(lambda_matrix + X.T.dot(X)).dot(X.T).dot(y)

    def predict(self, X):
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        return X.dot(self.coefs)
