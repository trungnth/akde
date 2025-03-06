import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from akde import akde

# Set random seed for reproducibility
np.random.seed(42)
n_samples = 1000

# Define 5 Gaussian mixture components (means, standard deviations, and weights)
means = [-4, -2, 0, 2, 4]       # Different means
stds = [0.5, 0.8, 0.3, 0.7, 1.0]  # Different standard deviations
weights = [0.2, 0.15, 0.25, 0.2, 0.2]  # Different weights (sum to 1)

# Generate samples from each Gaussian distribution
data = np.hstack([
    np.random.normal(mean, std, int(n_samples * weight))
    for mean, std, weight in zip(means, stds, weights)
]).reshape(-1, 1)  # Reshape to (n,1) for AKDE

# Perform adaptive kernel density estimation
pdf, meshgrids, bandwidth = akde(data)

# Reshape the PDF to match the shape of the grid
pdf = pdf.reshape(meshgrids[0].shape)

# Compute the true density function by summing individual Gaussians
true_pdf = np.sum([
    weight * norm.pdf(meshgrids[0], mean, std)
    for mean, std, weight in zip(means, stds, weights)
], axis=0)

# Plot the density estimation, true distribution, and histogram
plt.figure(figsize=(8, 5))
plt.hist(data, bins=50, density=True, alpha=0.3, color='gray', label="Histogram (normalized)")
plt.plot(meshgrids[0], pdf, label="AKDE", color='blue', linewidth=2)
plt.plot(meshgrids[0], true_pdf, label="True Distribution", color='red', linestyle='dashed', linewidth=2)
plt.xlabel("X")
plt.ylabel("Density")
plt.title("Histogram vs AKDE vs True Distribution")
plt.legend()
plt.savefig("1D-AKDE-Density.png", transparent=False, dpi=600, bbox_inches="tight")
plt.show()
