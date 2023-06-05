[![Alt text](/branding/logo1.png "Optional title")](https://github.com/Kacperaan/matrixforge)
[![DOI](https://zenodo.org/badge/630897036.svg)](https://zenodo.org/badge/latestdoi/630897036)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI](https://img.shields.io/pypi/v/hy)](https://pypi.org/project/matrixforge/)

> **Warning** MatrixForge is still in EARLY developement and may not work properly or have some others issues

# MatrixForge - Introduction
 
**MatrixForge** is a open-source Python library used building and training artificial neural networks. MatrixForge allows you to create and train neural networks in a simple and intuitive way, using built-in tools. The MatrixForge library was created to provide an easy and convenient way to build and train neural networks.

## Installation
To start using the MatrixForge library, install it via PyPI or Conda:

PyPI:
```python
pip install matrixforge
```
Conda:
> **Warning** The library is not yet published on Conda
```python
conda install -c anaconda matrixforge
```
## Usage
To start using the MatrixForge library, import the appropriate classes or modules. A simplest example of using the library looks like this:
```python
import matrixforge as mf

MODEL = mf.Model([
    mf.Input(nodes=2),
    mf.Layer(nodes=3, activation='selu'),
    mf.Layer(nodes=3, activation='softmax')
])

a = mf.modelCompute(bias=1, model=MODEL)
```
[Documentation](DOCUMENTATION.md)

[Contributing](.github/CONTRIBIUTING.md)
[a](/README.md#PyPI)
