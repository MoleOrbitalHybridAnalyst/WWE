import numpy as np
from statsmodels.tsa import ar_model

def clean_ts(ts, timestep=2.4188843e-5):
    ''' remove potentially duplicated timesteps due to restart '''
    def roundint(x):
        return np.vectorize(int)(np.round(x))
    time = ts[:,0]
    maxstep = roundint( (np.max(time) - time[0]) / timestep )
    ts_res = np.empty((maxstep+1, ts.shape[1]))
    ts_res.fill(np.nan)
    indices = roundint( (time - time[0]) / timestep )
    for i in range(ts.shape[1]):
        np.put_along_axis(ts_res[:,i], indices, ts[:,i], axis=0)
    if np.sum(np.isnan(ts_res)) != 0:
        print(f"WARNING: np.sum(np.isnan(ts_res)) = {np.sum(np.isnan(ts_res))}")
        ts_res[np.isnan(ts_res)] = np.mean(ts_res[~np.isnan(ts_res)])
    return ts_res

def sample_variance(noise_sigma, rho, n):
    var = noise_sigma**2 / (1 - rho**2)
    var /= (n**2)
    var *= (n + 2 * n * rho / (1 - rho) + 2 * (rho**n - 1) / (1 - rho)**2 * rho)
    return var

def get_arfit(ts, trim_stepsize=None):
    if trim_stepsize is not None:
        N = len(ts)
        assert trim_stepsize < N // 2
        minvar = np.inf
        min_t = None
        for i in range(max(1, N // trim_stepsize // 3)):
            m = ar_model.AutoReg(ts[i*trim_stepsize:,1], 1, seasonal=False)
            res = m.fit()
            delta_, rho_ = res.params
            sample_var = sample_variance(np.sqrt(res.sigma2), rho_, N)
            if sample_var < minvar:
                minvar = sample_var
                min_t = i
        ts_ = ts[min_t*trim_stepsize:]
    else:
        ts_ = ts
    m = ar_model.AutoReg(ts_[:,1], 1, seasonal=False)
    res = m.fit()
    return res, ts_

def process(ts, timestep=None, trim_stepsize=None, decorr_thres=None):
    if timestep is not None:
        ts = clean_ts(ts, timestep)
    if decorr_thres is None and trim_stepsize is None:
        return ts
    fitres, ts = get_arfit(ts, trim_stepsize)
    _, rho = fitres.params
    if decorr_thres is not None:
        assert decorr_thres > 0
        tau = int(np.ceil(
            np.log(
                decorr_thres * (1-rho**2) / np.sqrt(fitres.sigma2)
                ) / np.log(rho)))
    else:
        tau = 1
    newN = len(ts) // tau * tau
    return ts[-newN::tau]
