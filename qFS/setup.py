# setup.py for qFS package
from setuptools import setup, find_packages

setup(
    name="qFS",
    version="0.0.2",
    description="Quick Feature Selection: A fast correlation-based filter for efficient feature selection in machine learning",
    long_description="""
    qFS (Quick Feature Selection) is a Python package implementing fast correlation-based 
    filter algorithms for feature selection. The package provides three main algorithms:
    
    - qFS: Basic threshold-based feature selection using symmetrical uncertainty
    - qFSK: Select top K features with redundancy removal
    - qFSiP: Feature selection by dividing the feature space into pieces
    
    These algorithms are designed for efficient dimensionality reduction while 
    preserving the most informative features for machine learning tasks.
    
    Based on information theory metrics like entropy and symmetrical uncertainty.
    """,
    long_description_content_type="text/markdown",
    author="Ayushmaan Singh",
    url="https://github.com/Ayushmaan06/qFS",  
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scikit-learn"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
    ],
    python_requires='>=3.7'
)