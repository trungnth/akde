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

# KDE Visualization
Using contour plot or imshow for 2D data, isosurface or volume plot for 3D data


[matlab]: https://www.mathworks.com/matlabcentral/fileexchange/58312-kernel-density-estimator-for-high-dimensions
[paper]: https://dx.doi.org/10.1214/10-AOS799

