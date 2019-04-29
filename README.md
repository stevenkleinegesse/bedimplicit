# bedimplicit
Source code for "Efficient Bayesian Experimental Design for Implicit Models" - https://arxiv.org/abs/1810.09912

# Installation Instructions

The easiest way to install all the required packages is via Anaconda.

## Anaconda

Create an anaconda environment with the appropriate package list:

```
conda env create -n <env_name> --file req_conda.txt
```

Then activate the environment and install all the required pip files that are not available in the anaconda distribution:

```
conda activate <env_name>
pip install -r req_pip.txt
```

If that fails to work somehow, you can manually install the packages. Note that the environment needs to use Python 3.6 and **not 3.7**, as gpyopt currently fails to work with that version. All the required packages are:

- python v3.6
- numpy, scipy, matplotlib
- joblib v0.11
- glmnet
- gpyopt 
- scikit-learn

If you are using anaconda, manually install packages by doing

```
conda env create <env_name> python=3.6
conda activate <env_name>
conda install numpy scipy matplotlib
conda install joblib==0.11
pip install glmnet
pip install gpyopt
```