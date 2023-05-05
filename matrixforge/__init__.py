'''
MatrixForge
===========
Copyright (C) 2023 Kacper Popek 
All rights reserved(FOR MORE INFORMATIONS READ LICENCE)
'''

from numpy import *
import matplotlib.pyplot as plt

def relu(x):
    return maximum(0, x)
def sigmoid(x):
    return 1/(1+exp(-x))
def softplus(x):
    return log(1+e**x)
def softmax(x):
    e_x = exp(x - max(x))
    return e_x / e_x.sum(axis=0)
def tanhh(x):
    return tanh(x)
def d_relu(x):
    return 1. * (x > 0)
def d_sigmoid(x):
    a = 1/(1+e**(-x))
    return a*(1-a)
def d_tanh(x):
    return 1.-tanh(x)**2
def d_softplus(x):
    return 1/(1+e**(-x))
def d_softmax(x):
    soft = exp(x) / sum(exp(x), axis=0)
    return diag(soft) - outer(soft, soft)
def sgd(weights, gradient, learning_rate):
    for i in range(len(weights)):
        weights[i] = weights[i] - learning_rate * gradient[i]
    return weights
class createLayer:
    def __init__(self, nodes, activation=''):
        self.nodes = nodes
        self.activation = activation
        if activation == 'relu':
            self.activation = relu
            self.derivative = d_relu
        elif activation == 'sigmoid':
            self.activation = sigmoid
            self.derivative = d_sigmoid
        elif activation == 'softmax':
            self.activation = softmax
            self.derivative = d_softmax
        elif activation == 'softplus':
            self.activation = softplus
            self.derivative = d_softplus
        elif activation == 'tanh':
            self.activation = tanhh
            self.derivative = d_tanh
        else:
            raise ValueError("Unsupported activation function")
class createModel:
    def __init__(self, neurons):
        self.neurons = len(neurons)
        self.inputl = neurons[0]
        self.outputl = neurons[-1]
        if len(neurons) == 2:
            print('NWM')
        elif len(neurons) == 1:
            raise ValueError("Model must contain at least 2 layers, Input and Output")
        else:
         K = len(neurons)-2       
         middle = len(neurons) // 2
         self.hiddenl = [neurons[i] for i in range(middle - K//2, middle + (K//2) + 1) if i != 0 and i != len(neurons) - 1]
class forwardPropagation:
    def __init__(self, model, biasvalue):
        self.model = model
        self.biasvalue = biasvalue
        self.inputlayer = model.inputlayer
        self.inputr = model.inputlayer.activation(random.randn(model.inputlayer.nodes))
        self.inputweights = random.randn(model.inputlayer.nodes, model.hiddenlayer.nodes)
        self.hiddenweights = random.randn(model.hiddenlayer.nodes, model.outputlayer.nodes)
        self.input_activation = model.inputlayer.activation
        self.hidden_activation = model.hiddenlayer.activation
        self.output_activation = model.outputlayer.activation
        self.hiddenlayer = model.hiddenlayer
        self.outputlayer = model.outputlayer
        self.hiddenlfp = self.hidden_activation(biasvalue+dot(ones((1, self.inputlayer.nodes)), self.inputweights))
        self.outputlfp = self.output_activation(biasvalue+dot(self.hiddenlfp, self.hiddenweights))
class backPropagation:
    def __init__(self, model, expectedvalue, learning_rate):
        self.model = model
        self.neurons = model.inputr, model.hiddenlfp, model.outputlfp
        self.error = expectedvalue-model.outputlfp
        self.derivative = model.inputlayer.derivative, model.hiddenlayer.derivative, model.outputlayer.derivative
        self.weights = model.inputweights, model.hiddenweights
        self.learning_rate = learning_rate
        self.biasvalue = model.biasvalue
class modelTrain:
    def __init__(self,model,epochs):
        self.model = model
        self.epochs = epochs

def model_visualize(layer_sizes,neuron_size='',color=''):
        num_layers = len(layer_sizes)
        layer_scale = len(layer_sizes)/10
        layer_offset = len(layer_sizes)/2
        if num_layers == 1:
          raise ValueError("Number of layer must contain two")
        layer_positions = []
        for l in range(num_layers):
           num_neurons = layer_sizes[l]
           layer_y = linspace(0, 1, num_neurons + 2)[1:-1]
           layer_y *= layer_scale
           layer_y += 0.1
           layer_x = ones(num_neurons) * layer_offset
           layer_positions.append(column_stack((layer_x, layer_y)))
           layer_offset += 1
        connections = []
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        color_index = 0
        for l in range(num_layers - 1):
           positions1 = layer_positions[l]
           positions2 = layer_positions[l + 1]
           for i in range(positions1.shape[0]):
             for j in range(positions2.shape[0]):
                 connections.append(vstack((positions1[i], positions2[j])))
                 if color_index < len(colors):
                     color_index += 1
                 else:
                     color_index = 0
        fig, ax = plt.subplots()
        for pos in layer_positions:
         ax.scatter(pos[:, 0], pos[:, 1], color=color, s=neuron_size)
        for i in range(len(connections)):
         ax.plot(connections[i][:, 0], connections[i][:, 1], color=colors[i % len(colors)], linewidth=neuron_size/100)
         ax.axis('off')
        plt.show()
