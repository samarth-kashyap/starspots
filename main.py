import numpy as np
from starspot import Spot as SP
from twospot import twoSpotStar as ST
import starCycle as SC
from multiprocessing import Pool
import matplotlib.pyplot as plt
import emcee
import corner


KIC, p_rot = 3733735, 2.57
KIC, p_rot = 3448722, 0.4
KIC, p_rot = 2013754, 4.85
LC = SC.lightCurve(KIC=KIC, p_rot=p_rot)
lc_data = LC.data
lc_time = LC.daynum

max_time_idx = np.argmin(abs(lc_time - 5*p_rot))
lc_data = lc_data[:max_time_idx]
lc_data -= lc_data.mean()
lc_data /= abs(lc_data).max()
lc_time = lc_time[:max_time_idx]
time_step = (lc_time[1] - lc_time[0])/365.25

def get_model(theta):
    # area, inclination = theta
    area1, area2, alpha, inclination, lon1, lon2, lat1, lat2 = theta
    star = ST(inclination=inclination,
              no_evolution=True,
              max_area1=area1,
              max_area2=area2,
              Io=(abs(lc_data).max())**2,
              # Io=(abs(lc_data).max()),
              total_time=lc_time[-1]/365.25,
              time_step=time_step,
              time_arr=lc_time/365.25,
              prot=p_rot,
              lat1=np.radians(lat1), lat2=np.radians(lat2),
              lon1=np.radians(lon1), lon2=np.radians(lon2),
              alpha=alpha)
    star.simulate_spots()
    star.compute_light_curve()
    model = star.light_curve - star.light_curve.mean()
    return model


def log_likelihood(theta, x, y):
    model = get_model(theta)
    sigma = 20e-6
    logl = -0.5 * np.sum((y - model) ** 2 / sigma**2)
    return logl

def log_prior(theta):
    # area, inclination = theta
    # if (1e-3 < area < 9.9e5) and (0.0 < inclination < 90.0):
    #     return 0.0
    # area, inclination = theta
    area1, area2, alpha, inclination, lon1, lon2, lat1, lat2 = theta
    if ((1e-3 < area1 < 9.9e5) and
        (1e-3 < area2 < 9.9e5) and
        (0.0 < inclination < 90.0) and
        (0.1 < alpha < 1.0) and
        (-90. < lon1 < 270.) and
        (-90. < lon2 < 270.) and
        (-85. < lat1 < 85.) and
        (-85. < lat2 < 85.)):
        return 0.0
    return -np.inf

def log_probability(theta, x, y):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y)


# _star = ST(inclination=5., no_evolution=True, max_area1=1e3)
# _star.simulate_spots()
# _star.compute_light_curve()
# time = _star.time_arr*1.0
# lc_observed = _star.light_curve*1.0
# del _star

# pos = [0.9e1, 5] + 1e-1*np.random.randn(5, 2)
params = np.array([5e4, 3e3, 0.5, 5, 50, 60, -45, 55])
pos =  (np.random.randn(17, 8)/50. + 1)*params
nwalkers, ndim = pos.shape


with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool,
                                    args=(lc_time, lc_data))
    sampler.run_mcmc(pos, 1700, progress=True)

fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
# labels = ["area", "inclination"]
labels = ["area1", "area2", "alpha", "inclination", "lon1", "lon2", "lat1", "lat2"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number");

flat_samples = sampler.get_chain(discard=200, thin=15, flat=True)
print(flat_samples.shape)

fig = corner.corner(flat_samples, labels=labels) #, truths=[1e3, 5.])
