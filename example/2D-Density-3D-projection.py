import numpy as np
import plotly.graph_objects as go
from akde import akde

# Generate synthetic 2D mixture distribution data
np.random.seed(42)
n_samples = 10000

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

# AKDE
pdf, meshgrids, bandwidth = akde(X)

Z = pdf.reshape(meshgrids[0].shape)

fig = go.Figure(data=[
    go.Surface(
        x=meshgrids[0], y=meshgrids[1], z=Z,
        colorscale='Turbo',
        opacity=1,
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
])

# Update layout settings
fig.update_layout(
    title="",
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title="Density",
        xaxis=dict(tickmode='array', tickvals=np.arange(-10, 11, 2.0), showgrid=False),
        yaxis=dict(tickmode='array', tickvals=np.arange(-10, 11, 2.0), showgrid=False),
        zaxis=dict(tickmode='array', tickvals=np.arange(0, 0.6, 0.1), showgrid=False), 
    ),
    margin=dict(l=0, r=0, t=0, b=0)
)

fig.update_layout(scene = dict(
                    xaxis = dict(
                         backgroundcolor="rgb(255, 255, 255)",
                         gridcolor="white",
                         showbackground=True,
                         zerolinecolor="white",),
                    yaxis = dict(
                        backgroundcolor="rgb(255, 255,255)",
                        gridcolor="white",
                        showbackground=True,
                        zerolinecolor="white"),
                    zaxis = dict(
                        backgroundcolor="rgb(255, 255,255)",
                        gridcolor="white",
                        showbackground=True,
                        zerolinecolor="white",),)
                  )
  
fig.update_scenes(xaxis_visible=True, yaxis_visible=True,zaxis_visible=False)
fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", autosize=True)
fig.write_html("2D-AKDE-density-3D-projection.html")
