import numpy as np
import plotly.graph_objects as go
from akde import akde

# Set random seed for reproducibility
np.random.seed(12345)

# Number of samples
NUM = 10000

# Generate synthetic 3D data (mixture of 3 Gaussians)
means = [(2, 3, 1), (7, 7, 4), (3, 9, 8)]
std_devs = [(1.2, 0.8, 1.0), (1.5, 1.2, 1.3), (1.0, 1.5, 0.9)]
num_per_cluster = NUM // len(means)

# Create 2D numpy array data with shape: (NUM, 3)
data = np.vstack([
    np.random.multivariate_normal(mean, np.diag(std), num_per_cluster)
    for mean, std in zip(means, std_devs)
])  

# AKDE 
pdf, meshgrids, bandwidth = akde(data) 

# ====== PLOTTING THE 3D DENSITY WITH PLOTLY VOLUME OR ISOSURFACE PLOT ======== #

fig = go.Figure()

fig.add_trace(
    go.Volume(
        x=meshgrids[0].ravel(),
        y=meshgrids[1].ravel(),
        z=meshgrids[2].ravel(),
        value=pdf.ravel(),
        isomin=pdf.max()/50,
        opacity=0.2,
        surface_count=50,
        colorscale="Turbo",
        colorbar=dict(
            title="Density",
            titleside="right",
            titlefont=dict(size=14),
            thickness=15,
            len=0.6,  
            x=0.7,  
            y=0.5
        )
    )
)

# Layout customization
fig.update_layout(
    scene=dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z",
    ),
    margin=dict(l=0, r=0, b=0, t=0),
    autosize=True
)

# Export interactive plot in html, for export static figure install kaleido: pip install kaleido
fig.write_html("3d-density-akde-plotly.html")
# The 3D density plot from akde contains 128^3 points by default
# It is not recommended to show Plotly interactive plot in ipynb
