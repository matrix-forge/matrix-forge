 # MatrixForge - Documentation

MatrixForge is a library for building and training neural networks written in Python. This library allows you to create and train neural networks in a simple and intuitive way, using built-in tools. The MatrixForge library was created to provide an easy and convenient way to build and train neural networks.

## Library content
MatrixForge consists of the following modules:

- createLayer - a class that allows you to create layers of neural networks, where the user can define the number of nodes in a layer and the activation function for this layer.

- ***modelCreate*** - a class representing a neural network model, consisting of an input layer, one or more hidden layers and an output layer.

- ***forwardPropagation*** - a class that performs forward propagation for a neural network model.

- ***backPropagation*** - a class that implements backpropagation for the neural network model.

- ***modelTrain*** - a class that allows you to train a neural network model.

- ***modelArchitecture*** - a class that allows you to display the architecture of the created neural network model.

## Usage
To start using the MatrixForge library, import the appropriate classes or modules. A simplest example of using the library looks like this:
```
import matrixforge as mf

# Creating a neural network layers

inputlayer = mf.createLayer(nodes=3, activation='relu') 
hiddenlayer = mf.createLayer(nodes=3, activation='sigmoid')
outputlayer = mf.createLayer(nodes=2, activation='softmax')

# Creating a neural network model

model = mf.forwardPropagation(mf.modelCreate(inputlayer, hiddenlayer, outputlayer, hiddenlayeram=1), biasvalue=1)

# Training a neural network model

y = [[1, 0], [0, 1]]
model = mf.backPropagation(model=model, learning_rate=0, expectedvalue=y)

# Displaying the architecture of the neural network model

mf.modelArchitecture(model)
```
