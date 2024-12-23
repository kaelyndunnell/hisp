[![CI](https://github.com/kaelyndunnell/hisp/actions/workflows/ci.yml/badge.svg)](https://github.com/kaelyndunnell/hisp/actions/workflows/ci.yml)
[![docs](https://readthedocs.org/projects/hisp/badge/?version=latest)](https://hisp.readthedocs.io/en/latest/)

# HISP

Hydrogen Inventory Simulations for PFCs (HISP) is a series of code that uses FESTIM to simulate deuterium and tritium inventories in a fusion tokamak first wall and divertor PFCs. 

## How to Run:

Clone the repository:

```
git clone https://github.com/kaelyndunnell/hisp
cd hisp
```

Run this command to create a new environment with the right dependencies (eg. dolfinx):
```
conda env create -f environment.yml
```

Then, activate the environment:
```
conda activate hisp-env
```


Install the `hisp` package with:

```
python -m pip install -e .
```

This will also install the pip dependencies like `h-transport-materials` and `FESTIM`.

> **_NOTE:_**  Using `-e` with pip will install the package in editable mode, meaning the source code can be modified without having to reinstall the package.

## Run the tests

After cloning the repo and installing hisp, run:

```
python -m pytest test
```

> **_NOTE:_**  Make sure to install the test dependencies with `python -m pip install -e .[test]`
