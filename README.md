<div align="center">
 <a href="https://github.com/matrix-forge/matrix-forge"><img src="/branding/logo2-dark.png" width="17%"></img></a>
</div>
 
## <div align="center">A Python library for developing machine learning models</div>
<br>

<div align="center">
 
[![DOI](https://zenodo.org/badge/630897036.svg)](https://zenodo.org/badge/latestdoi/630897036)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI](https://img.shields.io/pypi/v/hy)](https://pypi.org/project/matrixforge/)
 </div>

> **Warning** MatrixForge is still in EARLY developement and may not work properly or have some others issues
> 
# MatrixForge - Introduction
 
**MatrixForge** is a open-source Python library used building and training artificial neural networks. MatrixForge allows you to create and train neural networks in a simple and intuitive way. The MatrixForge library was created to provide an easy and convenient way to develop neural networks for everyone.

## Installation
> **Note** Not every time the actualization will be on time on PyPI or Conda

To start using the MatrixForge library, install it via PyPI or Conda:

PyPI:
```python
$ pip install matrixforge
```
Conda:
> **Warning** The library is not yet published on Conda
```python
$ conda install -c anaconda matrixforge
```
## Usage
Example of the simplest MatrixForge model:
 ## Python
```python
>>> import matrixforge as mf

>>> MODEL = mf.Model([
>>> mf.Input(nodes=2),
>>> mf.Layer(nodes=3, activation='sigmoid'),
>>> mf.Layer(nodes=3, activation='softmax')])

>>> MODEL = mf.modelCompute(bias=1, model=MODEL)
```
[Documentation](DOCUMENTATION.md)

[Contributing](.github/CONTRIBIUTING.md)

[To Do](TODO.md)
