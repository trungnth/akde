import numpy as np
from scipy.linalg import cholesky

def akde(X, ng=None, grid=None, gam=None):
    n, d = X.shape
    MAX, MIN = np.max(X, axis=0), np.min(X, axis=0)
    scaling = MAX - MIN
    MAX = MAX + scaling / 10 
    MIN = MIN - scaling / 10
    scaling = MAX - MIN
    X = (X - MIN) / scaling

    if ng is None:
        ng = 100  # Default grid size if not provided
    if grid is None:
        linspaces = [np.linspace(MIN[i], MAX[i], ng) for i in range(d)]
        meshgrids = np.meshgrid(*linspaces)
        grid = np.vstack([mg.ravel() for mg in meshgrids]).T
    mesh = (grid - MIN) / scaling

    gam = gam or int(np.ceil(n ** 0.5))
    bandwidth = 0.1 / n ** (d / (d + 4))
    perm = np.random.permutation(n)
    mu = X[perm[:gam], :]
    w = np.random.rand(gam)
    w /= np.sum(w)
    Sig = np.random.rand(d, d, gam) * np.eye(d)[:, :, np.newaxis] * bandwidth
    ent = -np.inf

    for iter in range(1500):
        Eold = ent
        w, mu, Sig, bandwidth, ent = regEM(w, mu, Sig, bandwidth, X)
        err = abs((ent - Eold) / ent)
        if err < 1e-4 or iter > 1000:
            break

    pdf = probfun(mesh, w, mu, Sig) / np.prod(scaling)
    bandwidth *= scaling
    return pdf, meshgrids, bandwidth

def probfun(x, w, mu, Sig):
    gam, d = mu.shape
    pdf = np.zeros(x.shape[0])
    for k in range(gam):
        L = cholesky(Sig[:, :, k], lower=True)
        logpdf = -0.5 * np.sum((np.dot(x - mu[k, :], np.linalg.inv(L)) ** 2), axis=1) + np.log(w[k]) - np.sum(np.log(np.diag(L))) - d * np.log(2 * np.pi) / 2
        pdf += np.exp(logpdf)
    return pdf

def regEM(w, mu, Sig, bandwidth, X):
    gam, d = mu.shape
    n, d = X.shape
    log_lh, log_sig = np.zeros((n, gam)), np.zeros((n, gam))

    for i in range(gam):
        L = cholesky(Sig[:, :, i], lower=True)
        Xcentered = X - mu[i, :]
        xRinv = np.dot(Xcentered, np.linalg.inv(L))
        xSig = np.sum((xRinv @ np.linalg.inv(L.T)) ** 2, axis=1) + np.finfo(float).eps
        log_lh[:, i] = -0.5 * np.sum(xRinv ** 2, axis=1) - np.sum(np.log(np.diag(L))) + np.log(w[i]) - d * np.log(2 * np.pi) / 2 - 0.5 * bandwidth ** 2 * np.trace(np.dot(np.linalg.inv(L), np.linalg.inv(L.T)))
        log_sig[:, i] = log_lh[:, i] + np.log(xSig)

    maxll, maxlsig = np.max(log_lh, axis=1), np.max(log_sig, axis=1)
    p = np.exp(log_lh - maxll[:, None])
    psig = np.exp(log_sig - maxlsig[:, None])
    density, psigd = np.sum(p, axis=1), np.sum(psig, axis=1)
    logpdf, logpsigd = np.log(density) + maxll, np.log(psigd) + maxlsig
    p /= density[:, None]
    ent = np.sum(logpdf)
    w = np.sum(p, axis=0)

    for i in np.where(w > 0)[0]:
        mu[i, :] = np.dot(p[:, i], X) / w[i]
        Xcentered = (X - mu[i, :]) * np.sqrt(p[:, i])[:, None]
        Sig[:, :, i] = np.dot(Xcentered.T, Xcentered) / w[i] + bandwidth ** 2 * np.eye(d)

    w /= np.sum(w)
    curv = np.mean(np.exp(logpsigd - logpdf))
    bandwidth = 1 / (4 * n * (4 * np.pi) ** (d / 2) * curv) ** (1 / (d + 2))
    return w, mu, Sig, bandwidth, ent
