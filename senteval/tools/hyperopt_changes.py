'''
    From pyll_utils.py
'''

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import numpy as np
from scipy.special import erf
from builtins import str
from builtins import zip
from builtins import range
from past.builtins import basestring
from builtins import object
from functools import partial, wraps
from hyperopt.base import DuplicateLabel
from hyperopt.pyll.base import Apply
from hyperopt.pyll import scope
from hyperopt.pyll import as_apply
from hyperopt.tpe import normal_cdf
from hyperopt.pyll_utils import hyperopt_param, validate_label

EPS = 1e-12

def logsum_rows(x):
    R, C = x.shape
    m = x.max(axis=1)
    return np.log(np.exp(x - m[:, None]).sum(axis=1)) + m

@validate_label
def hp_logbaseuniform(label, *args, **kwargs):
    return scope.float(
        scope.hyperopt_param(label,
                             scope.logbaseuniform(*args, **kwargs)))


@validate_label
def hp_qlogbaseuniform(label, *args, **kwargs):
    return scope.float(
        scope.hyperopt_param(label,
                             scope.qlogbaseuniform(*args, **kwargs)))

@validate_label
def hp_logbasenormal(label, *args, **kwargs):
    return scope.float(
        scope.hyperopt_param(label,
                             scope.logbasenormal(*args, **kwargs)))


@validate_label
def hp_qlogbasenormal(label, *args, **kwargs):
    return scope.float(
        scope.hyperopt_param(label,
                             scope.qlogbasenormal(*args, **kwargs)))

"""
    From pyll/base.py
"""

@scope.define_pure
def logbase(a, base=np.exp(1)):
    return np.log(a)/np.log(base)


"""
    From pyll/stochastic.py
    Constructs for annotating base graphs.
"""

from builtins import range
from past.utils import old_div
import sys
from hyperopt.pyll.base import dfs, rec_eval, clone
from hyperopt.pyll.stochastic import implicit_stochastic, ERR, rng_from_seed


# -- BASE UNIFORM

@implicit_stochastic
@scope.define
def logbaseuniform(low, high, base=np.exp(1), rng=None, size=()):
    draw = rng.uniform(low, high, size=size)
    return np.exp(draw*np.log(base))

@implicit_stochastic
@scope.define
def qlogbaseuniform(low, high, q, base=np.exp(1), rng=None, size=()):
    draw = np.exp(rng.uniform(low, high, size=size)*np.log(base))
    return np.round(old_div(draw, q)) * q

# -- BASE NORMAL

@implicit_stochastic
@scope.define
def logbasenormal(mu, sigma, base=np.exp(1), rng=None, size=()):
    draw = rng.normal(mu, sigma, size=size)
    return np.exp(draw*np.log(base))


@implicit_stochastic
@scope.define
def qlogbasenormal(mu, sigma, q, base=np.exp(1), rng=None, size=()):
    draw = np.exp(rng.normal(mu, sigma, size=size)*np.log(base))
    return np.round(old_div(draw, q)) * q


'''
    From tpe.py
'''
from hyperopt.tpe import adaptive_parzen_sampler

@implicit_stochastic
@scope.define
def LBGMM1(weights, mus, sigmas, low=None, high=None, q=None, base=np.exp(1), rng=None, size=()):
    weights, mus, sigmas = list(map(np.asarray, (weights, mus, sigmas)))
    n_samples = np.prod(size)
    # n_components = len(weights)
    if low is None and high is None:
        active = np.argmax(
            rng.multinomial(1, weights, (n_samples,)),
            axis=1)
        assert len(active) == n_samples
        samples = np.exp(
            rng.normal(
                loc=mus[active],
                scale=sigmas[active])*np.log(base))
    else:
        # -- draw from truncated components
        # TODO: one-sided-truncation
        low = float(low)
        high = float(high)
        if low >= high:
            raise ValueError('low >= high', (low, high))
        samples = []
        while len(samples) < n_samples:
            active = np.argmax(rng.multinomial(1, weights))
            draw = rng.normal(loc=mus[active], scale=sigmas[active])
            if low <= draw < high:
                samples.append(np.exp(draw*np.log(base)))
        samples = np.asarray(samples)

    samples = np.reshape(np.asarray(samples), size)
    if q is not None:
        samples = np.round(old_div(samples, q)) * q
    return samples

@scope.define
def LBGMM1_lpdf(samples, weights, mus, sigmas, low=None, high=None, q=None, base=np.exp(1)):
    samples, weights, mus, sigmas = list(map(np.asarray,
                                         (samples, weights, mus, sigmas)))
    assert weights.ndim == 1
    assert mus.ndim == 1
    assert sigmas.ndim == 1
    _samples = samples
    if samples.ndim != 1:
        samples = samples.flatten()

    if low is None and high is None:
        p_accept = 1
    else:
        p_accept = np.sum(
            weights * (
                normal_cdf(high, mus, sigmas) -
                normal_cdf(low, mus, sigmas)))

    if q is None:
        # compute the lpdf of each sample under each component
        lpdfs = logbasenormal_lpdf(samples[:, None], mus, sigmas, base)
        rval = logsum_rows(lpdfs + np.log(weights))
    else:
        # compute the lpdf of each sample under each component
        prob = np.zeros(samples.shape, dtype='float64')
        for w, mu, sigma in zip(weights, mus, sigmas):
            if high is None:
                ubound = samples + old_div(q, 2.0)
            else:
                ubound = np.minimum(samples + old_div(q, 2.0), np.exp(high*np.log(base)))
            if low is None:
                lbound = samples - old_div(q, 2.0)
            else:
                lbound = np.maximum(samples - old_div(q, 2.0), np.exp(low*np.log(base)))
            lbound = np.maximum(0, lbound)
            # -- two-stage addition is slightly more numerically accurate
            inc_amt = w * logbasenormal_cdf(ubound, mu, sigma, base)
            inc_amt -= w * logbasenormal_cdf(lbound, mu, sigma, base)
            prob += inc_amt
        rval = np.log(prob) - np.log(p_accept)
    rval.shape = _samples.shape
    return rval


# -- Mixture of Log-Normals

@scope.define
def logbasenormal_cdf(x, mu, sigma, base=np.exp(1)):
    # wikipedia claims derived cdf (base changed) is
    # .5 + .5 erf( ln(x)/ln(base) - mu / sqrt(2 sigma^2))
    #
    # the maximum is used to move negative values and 0 up to a point
    # where they do not cause nan or inf, but also don't contribute much
    # to the cdf.
    if len(x) == 0:
        return np.asarray([])
    if x.min() < 0:
        raise ValueError('negative arg to logbasenormal_cdf', x)
    olderr = np.seterr(divide='ignore')
    try:
        top = old_div(np.log(np.maximum(x, EPS)), np.log(base)) - mu
        bottom = np.maximum(np.sqrt(2) * sigma, EPS)
        z = old_div(top, bottom)
        return .5 + .5 * erf(z)
    finally:
        np.seterr(**olderr)


@scope.define
def logbasenormal_lpdf(x, mu, sigma, base=np.exp(1)):
    # formula derived from wikipedia with base changed
    # http://en.wikipedia.org/wiki/Log-normal_distribution
    assert np.all(sigma >= 0)
    sigma = np.maximum(sigma, EPS)
    Z = sigma * x * np.log(base) * np.sqrt(2 * np.pi)
    E = 0.5 * (old_div((np.log(x)/np.log(base) - mu), sigma)) ** 2
    rval = -E - np.log(Z)
    return rval


@scope.define
def qlogbasenormal_lpdf(x, mu, sigma, q, base=np.exp(1)):
    # casting rounds up to nearest step multiple.
    # so lpdf is log of integral from x-step to x+1 of P(x)

    # XXX: subtracting two numbers potentially very close together.
    return np.log(
        logbasenormal_cdf(x, mu, sigma, base) -
        logbasenormal_cdf(x - q, mu, sigma, base))


# -- Uniform

@adaptive_parzen_sampler('logbaseuniform')
def ap_logbaseuniform_sampler(obs, prior_weight, low, high, base=np.exp(1),
                          size=(), rng=None):
    prior_mu = 0.5 * (high + low)
    prior_sigma = 1.0 * (high - low)
    weights, mus, sigmas = scope.adaptive_parzen_normal(
        scope.logbase(obs, base), prior_weight, prior_mu, prior_sigma)
    rval = scope.LBGMM1(weights, mus, sigmas, low=low, high=high, base=base,
                       size=size, rng=rng)
    return rval

@adaptive_parzen_sampler('qlogbaseuniform')
def ap_qlogbaseuniform_sampler(obs, prior_weight, low, high, q, base=np.exp(1),
                           size=(), rng=None):
    prior_mu = 0.5 * (high + low)
    prior_sigma = 1.0 * (high - low)
    weights, mus, sigmas = scope.adaptive_parzen_normal(
        scope.logbase(obs, base), prior_weight, prior_mu, prior_sigma)
        # scope.logbase(
        #     # -- map observations that were quantized to be below exp(low*ln(base))
        #     #    (particularly 0) back up to exp(low*ln(base)) where they will
        #     #    interact in a reasonable way with the AdaptiveParzen
        #     #    thing.
        #     scope.maximum(
        #         obs,
        #         scope.maximum(  # -- protect against exp(low*ln(base)) underflow
        #             EPS,
        #             scope.exp(low*np.log(base)))), base),
        # prior_weight, prior_mu, prior_sigma)
    return scope.LBGMM1(weights, mus, sigmas, low, high, q=q, base=base,
                       size=size, rng=rng)


# -- Normal

@adaptive_parzen_sampler('logbasenormal')
def ap_loglogbasenormal_sampler(obs, prior_weight, mu, sigma, base=np.exp(1), size=(), rng=None):
    weights, mus, sigmas = scope.adaptive_parzen_normal(
        scope.logbase(obs, base), prior_weight, mu, sigma)
    rval = scope.LBGMM1(weights, mus, sigmas, base=base, size=size, rng=rng)
    return rval

@adaptive_parzen_sampler('qlogbasenormal')
def ap_qlogbasenormal_sampler(obs, prior_weight, mu, sigma, q, base=np.exp(1), size=(), rng=None):
    log_obs = scope.logbase(scope.maximum(obs, EPS), base)
    weights, mus, sigmas = scope.adaptive_parzen_normal(
        log_obs, prior_weight, mu, sigma)
    rval = scope.LBGMM1(weights, mus, sigmas, q=q, base=base, size=size, rng=rng)
    return rval

"""
    To plot fake posterior.
"""
import matplotlib.pyplot as plt
import seaborn as sns
from hyperopt.pyll.stochastic import sample

def PosteriorPlot(space_search, trials, Nsamples=1000):
    # create dic of experiences
    d = {}
    for k in space_search.keys():
        d[k] = np.array([ (t['misc']['vals'][k][0], t['result']['loss']) for t in trials.trials])

    # create dic of samples from prior
    samples = {}
    for k in d.keys():
        samples[k] = [sample(space_search)[k] for x in range(100)]

    # plot prior and posterior
    # inspired by https://github.com/MBKraus/Hyperopt/blob/master/Hyperopt.ipynb
    # https://www.codementor.io/mikekraus/using-bayesian-optimisation-to-reduce-the-time-spent-on-hyperparameter-tuning-tgc3ikmp2
    for k in d.keys():
        f, ax = plt.subplots(figsize=(10,6))
        sns.set_palette("husl")
        sns.despine()
        ax = sns.kdeplot(np.array(samples[k]), label = 'Prior', linewidth = 3)
        ax = sns.kdeplot(d[k][:, 0], label = 'Posterior (as complete path)', linewidth = 3)
        ax.set_ylabel('Density', fontsize=12, fontweight='bold')
        ax2 = ax.twinx()
        ax2.scatter(d[k][:, 0], d[k][:, 1], c='blue', label='Loss ind. value')
        ax2.set_ylabel('loss', fontsize=12, fontweight='bold', color='blue')
        plt.title(k, fontsize=18, fontweight='bold')
        plt.xlabel(k, fontsize=12, fontweight='bold')
        plt.legend()
        #plt.setp(ax.get_legend().get_texts(), fontsize='12', fontweight='bold')
        plt.plot()


import matplotlib.pyplot as plt
import seaborn as sns
from hyperopt.pyll.stochastic import sample
def PosteriorPlot(space_search, trials, Nsamples=1000):
    # create dic of experiences
    d = {}
    for k in space_search.keys():
        d[k] = np.array([ (t['misc']['vals'][k][0], t['result']['loss']) for t in trials.trials])

    # create dic of samples from prior
    samples = {}
    for k in d.keys():
        samples[k] = [sample(space_search)[k] for x in range(100)]

    # plot prior and posterior
    # inspired by https://github.com/MBKraus/Hyperopt/blob/master/Hyperopt.ipynb
    # https://www.codementor.io/mikekraus/using-bayesian-optimisation-to-reduce-the-time-spent-on-hyperparameter-tuning-tgc3ikmp2
    for k in d.keys():
        f, ax = plt.subplots(figsize=(10,6))
        sns.set_palette("husl")
        sns.despine()
        ax = sns.kdeplot(np.array(samples[k]), label = 'Prior', linewidth = 3)
        ax = sns.kdeplot(d[k][:, 0], label = 'Posterior (as complete path)', linewidth = 3)
        ax.set_ylabel('Density', fontsize=12, fontweight='bold')
        ax2 = ax.twinx()
        ax2.scatter(d[k][:, 0], d[k][:, 1], c='blue', label='Loss ind. value')
        ax2.set_ylabel('loss', fontsize=12, fontweight='bold', color='blue')
        plt.title(k, fontsize=18, fontweight='bold')
        plt.xlabel(k, fontsize=12, fontweight='bold')
        plt.legend()
        #plt.setp(ax.get_legend().get_texts(), fontsize='12', fontweight='bold')
        plt.plot()
        