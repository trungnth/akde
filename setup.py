from setuptools import setup, find_packages

setup(
    name="akde",
    version="1.0.0",
    author="Trung Nguyen",
    author_email="trungnth@dnri.vn",
    description="Multivariate Adaptive Kernel Density Estimation via Gaussian Mixture Model",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/trungnth/akde",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
