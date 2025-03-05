# AKDE: Adaptive Kernel Density Estimation

AKDE provides a fast, adaptive kernel density estimator based on the Gaussian Mixture Model for multidimensional data. The [original MATLAB implementation][matlab] by Zdravko Botev does not appear to reference the algorithm described in the [corresponding paper][paper]. This Python re-implementation includes automatic grid construction for arbitrary dimensions and provides a detailed explanation of the method.

# Installation

You can install it via pip:  
```
pip install akde
```
# Usage
Providing any data in n by d numpy array (n rows, d columns)
```
from akde import akde
pdf, meshgrids, bandwidth = akde(data)

```
# 3D data

![3D Data Density Plot](https://raw.githubusercontent.com/trungnth/akde/refs/heads/main/media/3d-density.png)

# KDE Visualization
Using contour plot or imshow for 2D data, isosurface or volume plot for 3D data


[matlab]: https://www.mathworks.com/matlabcentral/fileexchange/58312-kernel-density-estimator-for-high-dimensions
[paper]: https://dx.doi.org/10.1214/10-AOS799

# AKDE: Apdaptive Multivariate Kernel Density Estimation via Gaussian Mixture Model

Given a dataset $X = \{x_1, x_2, \dots, x_n\}$ in $\mathbb{R}^d$, GMM models the density $f(x)$ of $X$ as a mixture of $K$ Gaussian components, expressed as:

$$
f(x) = \sum_{k=1}^K w_k \phi(x; \mu_k, \Sigma_k),
$$

where $w_k$ are non-negative mixture weights satisfying $\sum_{k=1}^K w_k = 1$. The Gaussian components, $\phi(x; \mu_k, \Sigma_k)$, are defined as:

$$
\phi(x; \mu_k, \Sigma_k) = \frac{1}{(2\pi)^{d/2} |\Sigma_k|^{1/2}} \exp\left(-\frac{1}{2}(x - \mu_k)^\top \Sigma_k^{-1} (x - \mu_k)\right).
$$

Here, $\mu_k \in \mathbb{R}^d$ and $\Sigma_k \in \mathbb{R}^{d \times d}$ are the mean vector and covariance matrix of the $k$-th component and $d$ is the dimensionality of the data. GMMs are particularly well-suited for AKDE, as they combine parametric modeling's structure with the flexibility required for adaptive density estimation.

The parameters of the GMM—weights $w_k$, means $\mu_k$, and covariance matrices $\Sigma_k$—are optimized using the EM algorithm, which alternates between the *E-step* and *M-step*. The EM algorithm maximizes the log-likelihood of the observed data. For a dataset $X = \{x_1, x_2, \dots, x_n\}$, the log-likelihood of the GMM is given by:

$$
\mathcal{L}(\Theta; X) = \sum_{i=1}^n \log\left(\sum_{k=1}^K w_k \phi(x_i; \mu_k, \Sigma_k)\right),
$$

where $\Theta = \{w_k, \mu_k, \Sigma_k\}_{k=1}^K$ represents the model parameters.

The algorithm begins by normalizing the input data to fit within a unit hypercube $[0, 1]^d$ for numerical stability. It then initializes $K$ Gaussian components by randomly selecting data points as initial means $\mu_k$. The covariance matrices $\Sigma_k$ are initialized as scaled identity matrices $I$, $\Sigma_k=h^2 \cdot I$ where the scaling $h$ is determined by an initial bandwidth derived from Silverman’s rule of thumb  $h = 0.1 / n^{d / (d + 4)}$ and the weights $w_k$ are assigned random values summing to one. These steps ensure a reasonable starting point for optimization.

In the *E-step*, the algorithm computes the responsibilities $\gamma_{ik}$, which measure the probability that the $k$-th component generated the $i$-th data point $x_i$. This is given by:

$$
\gamma_{ik} = \frac{w_k \phi(x_i; \mu_k, \Sigma_k)}{\sum_{j=1}^K w_j \phi(x_i; \mu_j, \Sigma_j)}.
$$

In the *M-step*, the model parameters are updated using the responsibilities:

$$
w_k = \frac{1}{n} \sum_{i=1}^n \gamma_{ik}, \quad
\mu_k = \frac{\sum_{i=1}^n \gamma_{ik} x_i}{\sum_{i=1}^n \gamma_{ik}}, \quad
\Sigma_k = \frac{\sum_{i=1}^n \gamma_{ik} (x_i - \mu_k)(x_i - \mu_k)^\top}{\sum_{i=1}^n \gamma_{ik}}.
$$

These updates ensure that the log-likelihood of the data increases with each iteration. The EM process is implemented in the `regEM` function in `akde.py`, which terminates when the change in log-likelihood is below a predefined tolerance.

After optimizing the GMM parameters, the algorithm evaluates the estimated density function $f(x)$ over a grid of points to enable visualization. This evaluation requires efficient computation of the Gaussian components for large datasets, which is achieved using Cholesky decomposition to factorize each covariance matrix $\Sigma_k$ as:

$$
\Sigma_k = L_k L_k^\top,
$$

where $L_k$ is a lower triangular matrix. This decomposition simplifies several key operations:

$$
|\Sigma_k| = \prod_{i=1}^d (L_k[i, i])^2,
$$

$$
\Sigma_k^{-1} = (L_k^\top)^{-1} L_k^{-1},
$$

$$
(x - \mu_k)^\top \Sigma_k^{-1} (x - \mu_k) = \|L_k^{-1} (x - \mu_k)\|^2.
$$

These computations, implemented in the `probfun` function, are critical for the efficient and stable evaluation of the GMM.

A unique feature of AKDE is the dynamic adjustment of bandwidths for each Gaussian component based on the local curvature of the density. The curvature $\text{curv}$ is defined as:

$$
\text{curv} = \int \|\nabla^2 f(x)\|_F^2 f(x) \, dx,
$$

where $\|\nabla^2 f(x)\|_F^2$ is the squared Frobenius norm of the Hessian matrix $\nabla^2 f(x)$, given by:

$$
\|\nabla^2 f(x)\|\_F^2 = \sum_{i=1}^d \sum_{j=1}^d \left(\frac{\partial^2 f(x)}{\partial x_i \partial x_j}\right)^2.
$$

The bandwidth $h_k$ for each component is then computed as:

$$
h_k = \left(\frac{1}{4n \pi^{d/2} |I|^{1/2} \cdot \text{curv}}\right)^{1/(d+2)},
$$

where $n$ is the number of data points, $d$ is the dimensionality, and $I$ is the identity matrix. Smaller bandwidths are assigned to regions of high curvature, improving the local adaptivity of the density estimate.

Entropy is used as a criterion to ensure the smoothness and generality of the density estimate. The entropy of the estimated density $f(x)$ is defined as:

$$
H = -\int f(x) \log f(x) \, dx.
$$

The algorithm maximizes entropy iteratively, promoting a smooth and unbiased density estimate while avoiding overfitting.

The `akde.py` implementation demonstrates the power of Gaussian Mixture Models in adaptive kernel density estimation. By combining the EM algorithm for parameter optimization, curvature-based bandwidth selection, and entropy maximization, the algorithm achieves flexible and accurate density estimation. The use of Cholesky decomposition ensures computational efficiency and stability, making the algorithm scalable for large datasets and high-dimensional problems. This seamless integration of mathematical rigor and computational efficiency highlights the versatility of GMMs in density estimation.

