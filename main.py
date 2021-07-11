import numpy as np
from starspot import Star as ST
from starspot import Spot as SP
from multiprocessing import Pool
import matplotlib.pyplot as plt
import emcee
import corner

def log_likelihood(theta, x, y):
    area, inclination = theta
    star = ST(inclination=inclination,
              no_evolution=True,
              max_spot_area=area)
    star.simulate_spots()
    star.compute_light_curve()
    model = star.light_curve
    sigma = 1e-6
    logl = -0.5 * np.sum((y - model) ** 2 / sigma**2)
    return logl

def log_prior(theta):
    area, inclination = theta
    if (1e1 < area < 1e4) and (0.0 < inclination < 10.0):
        return 0.0
    return -np.inf

def log_probability(theta, x, y):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y)


_star = ST(inclination=5., no_evolution=True, max_spot_area=1e3)
_star.simulate_spots()
_star.compute_light_curve()
time = _star.time_arr*1.0
lc_observed = _star.light_curve*1.0
del _star

pos = [0.9e3, 5] + 1e-1*np.random.randn(5, 2)
nwalkers, ndim = pos.shape


with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool,
                                    args=(time, lc_observed))
    sampler.run_mcmc(pos, 500, progress=True)

fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = ["area", "inclination"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number");

flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
print(flat_samples.shape)

fig = corner.corner(flat_samples, labels=labels, truths=[1e3, 5.])
