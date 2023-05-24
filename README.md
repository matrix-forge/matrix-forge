[![Alt text](/branding/logo1.png "Optional title")](https://github.com/Kacperaan/matrixforge)
[![DOI](https://zenodo.org/badge/630897036.svg)](https://zenodo.org/badge/latestdoi/630897036)
[![License](https://img.shields.io/badge/License-MIT-blue)](https://github.com/Kacperaan/MatrixForge/blob/master/LICENCE.txt)
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

# Creating a neural network layers
inputlayer = mf.createLayer(nodes=4, activation='relu') 
hiddenlayer = mf.createLayer(nodes=3, activation='sigmoid')
outputlayer = mf.createLayer(nodes=1, activation='softmax')

# Creating a neural network model
model = mf.forwardPropagation(mf.modelCreate(inputlayer, hiddenlayer, outputlayer, hiddenlayeram=1), biasvalue=1)

# Training a neural network model
y = [[1, 0], [0, 1]]
model = mf.backPropagation(model=model, learning_rate=0, expectedvalue=y)

# Displaying the architecture of the neural network model
mf.modelArchitecture(model)
```
## Contributing
[Documentation](.github/Documentation.md)
[Contributing](.github/CONTRIBIUTING.md)
