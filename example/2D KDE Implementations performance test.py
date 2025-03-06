import numpy as np
import matplotlib.pyplot as plt
from kde_diffusion import kde2d
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity
from akde import akde
from scipy.spatial.distance import jensenshannon
from scipy.interpolate import RegularGridInterpolator

# Set random seed for reproducibility
np.random.seed(12345)

# Number of samples 
NUM = int(20000)

# Generate Penta-Modal 2D Data (Five Gaussian Distributions)
means = [(2, 3), (6, 6), (10, 2), (4, 9), (8, 8)]
std_devs = [(1.0, 0.8), (1.5, 1.2), (1.2, 0.6), (1.3, 1.0), (1.1, 0.9)]
num_per_cluster = NUM // len(means)

data_x, data_y = [], []

for (mx, my), (sx, sy) in zip(means, std_devs):
    x = np.random.normal(mx, sx, num_per_cluster)
    y = np.random.normal(my, sy, num_per_cluster)
    data_x.append(x)
    data_y.append(y)

# Merge all clusters into a single dataset
data_x = np.concatenate(data_x)
data_y = np.concatenate(data_y)
data = np.vstack([data_x, data_y]).T

# Define a common grid for evaluation (100x100)
x_grid = np.linspace(data_x.min() - 1, data_x.max() + 1, 100)
y_grid = np.linspace(data_y.min() - 1, data_y.max() + 1, 100)
X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)

# Compute True PDF (Penta-Modal Gaussian Mixture)
true_pdf = np.zeros_like(X_mesh)
for (mx, my), (sx, sy) in zip(means, std_devs):
    true_pdf += (
        np.exp(-((X_mesh - mx) ** 2) / (2 * sx**2)) / np.sqrt(2 * np.pi * sx**2) *
        np.exp(-((Y_mesh - my) ** 2) / (2 * sy**2)) / np.sqrt(2 * np.pi * sy**2)
    )

# Normalize the true PDF to sum to 1
true_pdf /= np.sum(true_pdf)

# ======================== KDE IMPLEMENTATIONS ======================== #

kde_results = {}
mse_scores = {}
kl_scores = {}
js_scores = {}

# KDE-Diffusion (`kde2d`)

density_2D, grid, bandwidth = kde2d(data_x, data_y)
x_kde, y_kde = np.unique(grid[0]), np.unique(grid[1]) 
density_2D = (density_2D.T) / density_2D.sum()  
interp_func = RegularGridInterpolator((x_kde, y_kde), density_2D.T, method='linear', bounds_error=False, fill_value=0)
kde_results["KDE-Diffusion"] = interp_func((X_mesh, Y_mesh))


# SciPy KDE

scipy_kde = gaussian_kde(data.T, bw_method="silverman")
density_2D = scipy_kde(np.vstack([X_mesh.ravel(), Y_mesh.ravel()])).reshape(100, 100)
density_2D /= np.sum(density_2D) 
kde_results["SciPy KDE"] = density_2D


# Scikit-learn KDE

sklearn_kde = KernelDensity(kernel="gaussian", bandwidth='silverman').fit(data)
log_dens = sklearn_kde.score_samples(np.vstack([X_mesh.ravel(), Y_mesh.ravel()]).T)
density_2D = np.exp(log_dens).reshape(100, 100)
density_2D /= np.sum(density_2D) 
kde_results["Scikit-learn KDE"] = density_2D


# AKDE 

akde_pdf, meshgrids, _ = akde(data)  
x_akde, y_akde = np.unique(meshgrids[0]), np.unique(meshgrids[1]) 
density_2D = akde_pdf.reshape(meshgrids[0].shape)
density_2D /= np.sum(density_2D) 
interp_func = RegularGridInterpolator((x_akde, y_akde), density_2D.T, method='linear', bounds_error=False, fill_value=0)
kde_results["AKDE"] = interp_func((X_mesh, Y_mesh))


# ======================== EVALUATE KDE ACCURACY ======================== #

for method, kde_density in kde_results.items():
    mse_scores[method] = np.mean((kde_density - true_pdf) ** 2)
    kde_density = np.clip(kde_density, 1e-10, None)  # Avoid log(0)
    kl_scores[method] = np.sum(true_pdf * np.log(true_pdf / kde_density))
    js_scores[method] = jensenshannon(true_pdf.ravel(), kde_density.ravel())

# ======================== MULTIPANEL PLOTTING WITH METRICS ======================== #

fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True, constrained_layout=True)
axes = axes.ravel()

for i, (method, density_2D) in enumerate(kde_results.items()):
    ax = axes[i]
    im = ax.imshow(
        density_2D, extent=[x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()],
        origin="lower", cmap="turbo", aspect="auto"
    )
    
    # Scatter plot of a subset of the data points
    ax.scatter(data_x[::100], data_y[::100], color="white", s=2, alpha=0.6, label="True PDF Samples")

    # Title with KDE method and metrics
    
    mse = mse_scores[method]
    kl = kl_scores[method]
    js = js_scores[method]

    ax.set_title(f"{method}\nMSE: {mse:.6f} | KL: {kl:.6f} | JS: {js:.6f}", fontsize=10)
    ax.set_xlim(x_grid.min(), x_grid.max())
    ax.set_ylim(y_grid.min(), y_grid.max())
    ax.grid(True, linestyle="--", alpha=0.5)

    if i == 0:
        ax.legend(fontsize=10)

fig.suptitle("2D KDE Implementations with Performance Metrics", fontsize=14)
fig.supxlabel("X", fontsize=12)
fig.supylabel("Y", fontsize=12)

cbar = fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.02, pad=0.04)
cbar.set_label("Density")

plt.savefig("2D_KDEs_with_Metrics.png", transparent=False, dpi=600, bbox_inches="tight")
plt.savefig("2D_KDEs_with_Metrics.pdf", transparent=False, dpi=600, bbox_inches="tight")
plt.show()
