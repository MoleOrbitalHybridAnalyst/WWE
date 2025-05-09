import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = np

class WHAMSolver:

    def __init__(self, eps, maxiter=10000):
        self.eps = eps
        self.maxiter = maxiter

    def _get_pl(self, ni, nl, fi, cil):
        pl = nl / cp.einsum('i,i,il->l', ni, fi, cil) + 1e-200
        return pl / cp.sum(pl)

    def _get_fi(self, pl, cil):
        return 1 / cp.dot(cil, pl)

    def _update_pl(self, pl, ni, nl, cil):
        fi = self._get_fi(pl, cil)
        return self._get_pl(ni, nl, fi, cil)

    def kernel(self, nil, cil, init_fi=None):
        '''
        nil: histogram counts of simulation i in bin l
        cil: reweighting factors such that pil \propto cil pl
        '''
        nil = cp.asarray(nil)
        cil = cp.asarray(cil)
        if init_fi is None:
            fi = cp.ones(len(nil))
        else:
            fi = cp.asarray(init_fi)
        nl = cp.sum(nil, axis=0)
        ni = cp.sum(nil, axis=1)

        pl = self._get_pl(ni, nl, fi, cil)
        for i in range(self.maxiter):
            new_pl = self._update_pl(pl, ni, nl, cil)
            if cp.abs(cp.log(new_pl/ pl)).max() < self.eps:
                break
            else:
                pl = new_pl
        if i == self.maxiter - 1:
            raise Exception("not converged")

        return pl, self._get_fi(pl, cil)

class WHAMGaussianError:
    def __init__(self, solver):
        self.solver = solver

    def _get_hessian(self, pl, ni, cil):
        vil = cp.einsum('il,l->il', cil, pl)
        fi = 1. / cp.sum(vil, axis=1)
        res = cp.einsum('i,ix,iy->xy', ni * fi**2, vil, vil)
        diag = cp.einsum('i,il->l', ni * fi, vil)
        res = res - cp.diag(diag)
        return res

    def kernel(self, nil, cil, init_fi=None):
        raise NotImplementedError

class WHAMStatistic(WHAMGaussianError):
    def _get_log_posterior(self, pl, ni, nl, cil):
        fi = self.solver._get_fi(pl, cil)
        res = cp.dot(ni, cp.log(fi))
        res += cp.dot(nl, cp.log(pl))
        return res

    def _get_gaussian_log_posterior(self, gl, gmean, w, v):
        diff = gl - gmean
        diff = cp.einsum('li,l->i', v, diff)
        return cp.einsum('i,i,i->', diff, w, diff) / 2

    def kernel(self, nil, cil, init_fi=None, nsamples=1000):
        nil = cp.asarray(nil)
        cil = cp.asarray(cil)
        nl = cp.sum(nil, axis=0)
        ni = cp.sum(nil, axis=1)

        # wham solution
        pl0, fi = self.solver.kernel(nil, cil, init_fi=init_fi)
        self.pl = pl0
        self.fi = fi

        # log posterior hess w.r.t. free energies
        hess = self._get_hessian(pl0, ni, cil)

        w, v = cp.linalg.eigh(hess)
        mask = cp.abs(w) < 1e-10
        assert cp.sum(mask) == 1

        w_inv = cp.zeros_like(w)
        w_inv[~mask] = 1 / w[~mask]
        cov = -cp.einsum('ui,i,vi->uv', v, w_inv, v)

        # g = log(p)
        gmean = cp.log(pl0)
        gsamples = cp.random.multivariate_normal(
                gmean, cov, size=nsamples,
                check_valid='raise', method='eigh')

        # metropolis
        oldg = gmean
        samples = list()
        for newg in gsamples:
            logp =  self._get_log_posterior(cp.exp(newg), ni, nl, cil)
            logp -= self._get_log_posterior(cp.exp(oldg), ni, nl, cil)
            logp += self._get_gaussian_log_posterior(oldg, gmean, w, v)
            logp -= self._get_gaussian_log_posterior(newg, gmean, w, v)
            pacc = cp.exp(logp)
            if cp.random.uniform() < pacc:
                newp = cp.exp(newg)
                newp = newp / cp.sum(newp)
                samples.append(cp.log(newp))
                oldg = samples[-1]
            else:
                samples.append(oldg)

        return cp.exp(cp.asarray(samples))
