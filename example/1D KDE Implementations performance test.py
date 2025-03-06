# Comparison of 1D KDE Implementations with Performance Metrics
# Install dependencies: pip install kalepy kde_diffusion kdepy 
import numpy as np
import matplotlib.pyplot as plt
import kalepy as kale
from KDEpy import FFTKDE
from kde_diffusion import kde1d
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity
from akde import akde
from scipy.spatial.distance import jensenshannon

# Set random seed for reproducibility
np.random.seed(12345)

# Number of samples
NUM = int(1e4)

# Generate data: Mixture of normal and log-normal distributions
d1 = np.random.normal(4.0, 1.0, NUM)
d2 = np.random.lognormal(0, 0.5, NUM)
data = np.concatenate([d1, d2])

# Define True PDF
xx = np.linspace(0.0, 7.0, 100)[1:]  # Avoid zero for log-normal stability
yy = 0.5 * np.exp(-(xx - 4.0) ** 2 / 2) / np.sqrt(2 * np.pi)
yy += 0.5 * np.exp(-np.log(xx) ** 2 / (2 * 0.5 ** 2)) / (0.5 * xx * np.sqrt(2 * np.pi))

# ======================== KDE Implementations ======================== #

mse_scores = {}
kl_scores = {}
js_scores = {}

# KDE-Diffusion

density, grid, bandwidth = kde1d(data)


# Resample KDE-Diffusion to `xx`
density_interp = np.interp(xx, grid, density)
density_interp /= np.trapezoid(density_interp, xx) 


# Kalepy KDE

points, kdensity = kale.density(data, probability=True)

# KDEpy (Improved Sheather Jones - ISJ)

x, y_kdepy = FFTKDE(kernel='gaussian', bw='ISJ').fit(data).evaluate()

# SciPy KDE

scipy_kde = gaussian_kde(data, bw_method='silverman')
y_scipy = scipy_kde(xx)


# Scikit-learn KDE

sklearn_kde = KernelDensity(kernel='gaussian', bandwidth='silverman').fit(data[:, None])
log_dens = sklearn_kde.score_samples(xx[:, None])
y_sklearn = np.exp(log_dens)


# AKDE (Adaptive Kernel Density Estimation)

data_reshaped = data.reshape(-1, 1)  # AKDE expects 2D numpy input
akde_pdf, akde_grid, akde_bandwidth = akde(data_reshaped)

# ======================== COMPUTE ACCURACY METRICS ======================== #

kde_methods = {
    "KDEpy ISJ": (x, y_kdepy, 'c'),
    "KDE-Diffusion": (xx, density_interp, 'b'),  
    "SciPy KDE": (xx, y_scipy, 'm'),
    "Scikit-learn KDE": (xx, y_sklearn, 'y'),
    "Kalepy KDE": (points, kdensity, 'k'),
    "AKDE": (akde_grid[0], akde_pdf, 'g'),
}

for method, (x_vals, y_vals, _) in kde_methods.items():
    # Resample KDE Estimate to match `xx` grid for fair comparison
    y_vals_interp = np.interp(xx, x_vals, y_vals)

    # Compute Metrics
    mse_scores[method] = np.mean((y_vals_interp - yy) ** 2)
    y_vals_clipped = np.clip(y_vals_interp, 1e-10, None)  # Avoid log(0) error
    kl_scores[method] = np.sum(yy * np.log(yy / y_vals_clipped))
    js_scores[method] = jensenshannon(yy, y_vals_interp)

# ======================== MULTIPANEL PLOTTING WITH METRICS ======================== #

fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
axes = axes.ravel()

for i, (method, (x_vals, y_vals, color)) in enumerate(kde_methods.items()):
    ax = axes[i]
    
    # Plot True PDF (Reference)
    ax.plot(xx, yy, 'r--', alpha=0.9, lw=2, label="True PDF")
    
    # Plot KDE Estimate
    ax.plot(x_vals, y_vals, color=color, lw=2, label=method)
    
    # Retrieve Metrics
    mse = mse_scores.get(method, np.nan)
    kl = kl_scores.get(method, np.nan)
    js = js_scores.get(method, np.nan)

    # Formatting
    ax.set_title(f"{method}\nMSE: {mse:.6f} | KL: {kl:.6f} | JS: {js:.6f}", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)

    # Show legend only in the first panel
    if i == 0:
        ax.legend(fontsize=10)

# Set common labels
fig.suptitle("1D KDE Implementations with Performance Metrics", fontsize=14)
fig.supxlabel("X", fontsize=12)
fig.supylabel("Density", fontsize=12)

# Adjust layout for readability
plt.tight_layout()

# Save Figures
plt.savefig("1D_KDEs_with_Metrics.pdf", transparent=False, dpi=600, bbox_inches="tight")
plt.savefig("1D_KDEs_with_Metrics.png", transparent=False, dpi=600, bbox_inches="tight")
# Show Plot
plt.show()
