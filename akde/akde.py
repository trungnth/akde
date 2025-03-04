import numpy as np
from scipy.linalg import cholesky

def akde(X, ng=None, grid=None, gam=None):

    # Validate input data type and shape
    if not isinstance(X, np.ndarray):
        raise TypeError("Input data X must be a NumPy array.")
    if X.ndim != 2:
        raise ValueError("Input data X must be a 2D NumPy array of shape (n, d).")
    if X.shape[0] < 2 or X.shape[1] < 1:
        raise ValueError("Input data X must have at least 2 samples and at least 1 feature.")
    # Get the number of samples (n) and dimensions (d)
    n, d = X.shape

    # Compute the range of data along each dimension
    MAX, MIN = np.max(X, axis=0), np.min(X, axis=0)
    scaling = MAX - MIN

    # Expand the range slightly to avoid boundary effects
    MAX = MAX + scaling / 10 
    MIN = MIN - scaling / 10
    scaling = MAX - MIN

    # Normalize data to be within [0,1] in each dimension
    X = (X - MIN) / scaling

    # Set default grid size if not provided

    # Set default grid size based on dimension `d`
    if ng is None:
        if d == 1:
            ng = 1024
        elif d == 2:
            ng = 512
        elif d == 3:
            ng = 128
        else:
            ng = 64  # Default for higher dimensions  

    # Generate a uniform grid if not provided
    if grid is None:
        linspaces = [np.linspace(MIN[i], MAX[i], ng) for i in range(d)]
        meshgrids = np.meshgrid(*linspaces)
        grid = np.vstack([mg.ravel() for mg in meshgrids]).T  # Convert meshgrid to list of points

    # Normalize the grid using the same scaling as the data
    mesh = (grid - MIN) / scaling

    # Set default number of clusters 
    # `gam` – Number of Gaussian components or number of GAussian Mixtures
    # A cost-accuracy tradeoff parameter, where `gam < n`. 
    # The default value is `gam = int(np.ceil(n ** 0.5))`. 
    # Larger values may improve accuracy but always decrease speed. 
    # To accelerate the computation, reduce the value of `gam`.
    gam = min(gam or int(np.ceil(n ** 0.5)), n - 1)

    # Set initial bandwidth using Silverman's rule of thumb
    bandwidth = 0.1 / n ** (d / (d + 4))

    # Randomly permute data indices and select `gam` cluster centers
    perm = np.random.permutation(n)
    mu = X[perm[:gam], :]

    # Initialize random weights and normalize them
    w = np.random.rand(gam)
    w /= np.sum(w)

    # Initialize covariance matrices (each as diagonal scaled identity)
    Sig = np.random.rand(d, d, gam) * np.eye(d)[:, :, np.newaxis] * bandwidth

    # Initialize entropy to negative infinity for convergence tracking
    ent = -np.inf

    # Expectation-Maximization loop (Maximum 1500 iterations)
    for iter in range(1500):
        Eold = ent  # Store old entropy value
        w, mu, Sig, bandwidth, ent = regEM(w, mu, Sig, bandwidth, X)  # Update parameters using EM
        err = abs((ent - Eold) / ent)  # Compute relative change in entropy
        
        # Stop if convergence criterion met
        if err < 1e-4 or iter > 1000:
            break

    # Compute probability density function over the mesh grid
    pdf = probfun(mesh, w, mu, Sig) / np.prod(scaling)

    # Scale the bandwidth back to the original data scale
    bandwidth *= scaling

    return pdf, meshgrids, bandwidth

def probfun(x, w, mu, Sig):
    """ Compute the probability density function at points x """
    gam, d = mu.shape
    pdf = np.zeros(x.shape[0])

    # Compute probability density at each point in `x`
    for k in range(gam):
        # Cholesky decomposition for efficient covariance matrix inversion
        L = cholesky(Sig[:, :, k], lower=True)

        # Compute log probability density function
        logpdf = (-0.5 * np.sum((np.dot(x - mu[k, :], np.linalg.inv(L)) ** 2), axis=1) 
                  + np.log(w[k]) - np.sum(np.log(np.diag(L))) - d * np.log(2 * np.pi) / 2)

        # Accumulate the densities (mixture model)
        pdf += np.exp(logpdf)

    return pdf

def regEM(w, mu, Sig, bandwidth, X):
    """ Expectation-Maximization algorithm for adaptive KDE """
    gam, d = mu.shape
    n, d = X.shape

    # Log-likelihood matrices for probability and sigma computation
    log_lh, log_sig = np.zeros((n, gam)), np.zeros((n, gam))

    # E-step: Compute probabilities for each data point
    for i in range(gam):
        L = cholesky(Sig[:, :, i], lower=True)  # Cholesky decomposition of covariance matrix
        Xcentered = X - mu[i, :]  # Center data points

        xRinv = np.dot(Xcentered, np.linalg.inv(L))  # Transform using inverse Cholesky
        xSig = np.sum((xRinv @ np.linalg.inv(L.T)) ** 2, axis=1) + np.finfo(float).eps  # Compute squared Mahalanobis distance

        # Compute log-likelihood
        log_lh[:, i] = (-0.5 * np.sum(xRinv ** 2, axis=1) - np.sum(np.log(np.diag(L))) 
                        + np.log(w[i]) - d * np.log(2 * np.pi) / 2 
                        - 0.5 * bandwidth ** 2 * np.trace(np.dot(np.linalg.inv(L), np.linalg.inv(L.T))))

        # Compute log-variance estimation
        log_sig[:, i] = log_lh[:, i] + np.log(xSig)

    # Compute normalization factors for numerical stability
    maxll, maxlsig = np.max(log_lh, axis=1), np.max(log_sig, axis=1)

    # Compute posterior probabilities
    p = np.exp(log_lh - maxll[:, None])
    psig = np.exp(log_sig - maxlsig[:, None])

    # Compute density and normalization factors
    density, psigd = np.sum(p, axis=1), np.sum(psig, axis=1)
    logpdf, logpsigd = np.log(density) + maxll, np.log(psigd) + maxlsig
    p /= density[:, None]  # Normalize probabilities
    ent = np.sum(logpdf)  # Compute entropy for convergence

    # M-step: Update weights
    w = np.sum(p, axis=0)

    # Update means and covariance matrices
    for i in np.where(w > 0)[0]:
        # Compute new means
        mu[i, :] = np.dot(p[:, i], X) / w[i]

        # Compute new covariance matrices
        Xcentered = (X - mu[i, :]) * np.sqrt(p[:, i])[:, None]
        Sig[:, :, i] = np.dot(Xcentered.T, Xcentered) / w[i] + bandwidth ** 2 * np.eye(d)

    # Normalize weights
    w /= np.sum(w)

    # Compute curvature to adjust bandwidth. This is a Monte Carlo-like approximation. 
    # Instead of explicitly computing second derivatives, 
    # log-likelihood ratios are used to approximate the local curvature.
    curv = np.mean(np.exp(logpsigd - logpdf))
    bandwidth = 1 / (4 * n * (4 * np.pi) ** (d / 2) * curv) ** (1 / (d + 2))

    return w, mu, Sig, bandwidth, ent
