![Alt text](/branding/logo1.png "Optional title")

________________________________________________________________________________________________
**WARNING:** THE FOLLOWING PACKAGE MAY NOT WORK PROPERLY BECAUSE OF ITS EARLY DEVELOPEMENT STAGE 
________________________________________________________________________________________________
# MatrixForge - Documentation
 
**MatrixForge** is a open-source multi-language framework used building and training artificial neural networks. MatrixForge allows you to create and train neural networks in a simple and intuitive way, using built-in tools. The MatrixForge library was created to provide an easy and convenient way to build and train neural networks.

## Library content
MatrixForge consists of the following modules:

- ***createLayer*** - a class that allows you to create layers of neural networks, where the user can define the number of nodes in a layer and the activation function for this layer.

- ***createModel*** - a class representing a neural network model, consisting of an input layer, one or more hidden layers and an output layer.

- ***forwardPropagation*** - a class that performs forward propagation for a neural network model.

- ***backPropagation*** - a class that implements backpropagation for the neural network model.

- ***modelTrain*** - a class that allows you to train a neural network model.

- ***model_visualize*** - a class that allows you to display the architecture of the created neural network model.

## Installation
To start using the MatrixForge library, install it via PyPI or Conda:

PyPI:
```python
pip install matrixforge
```
Conda:
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

## License
[MIT License](/LICENCE.txt)

## Author
[Author](/LICENCE.txt)
