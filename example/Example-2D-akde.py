import numpy as np
import matplotlib.pyplot as plt
from akde import akde

# Generate synthetic 2D mixture distribution data
np.random.seed(42)
n_samples = 1000

# Define mixture components
mean1, cov1 = [2, 2], [[0.5, 0.2], [0.2, 0.3]]
mean2, cov2 = [-2, -2], [[0.6, -0.2], [-0.2, 0.4]]
mean3, cov3 = [2, -2], [[0.4, 0], [0, 0.4]]

# Sample from Gaussian distributions
data1 = np.random.multivariate_normal(mean1, cov1, n_samples // 3)
data2 = np.random.multivariate_normal(mean2, cov2, n_samples // 3)
data3 = np.random.multivariate_normal(mean3, cov3, n_samples // 3)

# Combine data into a single dataset
X = np.vstack([data1, data2, data3])

# Perform adaptive kernel density estimation
pdf, meshgrids, bandwidth = akde(X)

# Reshape the PDF to match the shape of the grid
pdf = pdf.reshape(meshgrids[0].shape)

# Plot the density estimate
plt.figure(figsize=(8, 6))
plt.imshow(pdf, extent=[meshgrids[0].min(), meshgrids[0].max(),
                        meshgrids[1].min(), meshgrids[1].max()],
           origin='lower', cmap='turbo', aspect='auto')

plt.colorbar(label="Density")
plt.scatter(X[:, 0], X[:, 1], s=5, color='white', alpha=0.3, label="Data points")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("2D DATA DENSITY PLOT")
plt.legend()
plt.savefig("2D-AKDE-Density.png", transparent=False, dpi=600, bbox_inches="tight")
plt.show()
