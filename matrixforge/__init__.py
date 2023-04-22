from numpy import *
import pickle
from os import remove, path

def relu(x):
    return maximum(0, x)
def sigmoid(x):
    return 1/(1+exp(-x))
def softplus(x):
    return log(1+e**x)
def softmax(x):
    e_x = exp(x - max(x))
    return e_x / e_x.sum(axis=0)
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
def sgd(w, g, lr):
    for i in range(len(w)):
        w[i] = w[i] - lr * g[i]
    return w
def activation_derivative(activation_function):
    if activation_function == 'relu':
        return d_relu
    elif activation_function == 'sigmoid':
        return d_sigmoid
    elif activation_function == 'tanh':
        return d_tanh
    elif activation_function == 'softmax':
        return d_softmax
    elif activation_function == 'softplus':
        return d_softplus
    else:
        raise ValueError("Unsupported activation function.")
class Layer:
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
            self.activation = tanh
            self.derivative = d_tanh
        else:
            raise ValueError("Unsupported activation function")
class Model:
    def __init__(self, inputs, hiddens, outputs, hiddenc):
        self.inputl = inputs
        self.hiddenl = hiddens
        self.outputl = outputs
        self.hiddenc = hiddenc
class forwardPropagation:
    def __init__(self, model, bias):
        self.model = model
        self.bias = bias
        self.inputl = model.inputl
        self.inputr = model.inputl.activation(random.randn(model.inputl.nodes))
        self.inputw = random.randn(model.inputl.nodes, model.hiddenl.nodes)
        self.hiddenw = random.randn(model.hiddenl.nodes, model.outputl.nodes)
        self.input_activation = model.inputl.activation
        self.hidden_activation = model.hiddenl.activation
        self.output_activation = model.outputl.activation
        self.hiddenl = model.hiddenl
        self.outputl = model.outputl
        self.hiddenlfp = self.hidden_activation(bias+dot(ones((1, self.inputl.nodes)), self.inputw))
        self.outputlfp = self.output_activation(bias+dot(self.hiddenlfp, self.hiddenw))
class backPropagation:
    def __init__(self,model,error,learning_rate):
        self.model = model
        self.neurons = model.inputr,model.hiddenlfp,model.outputlfp
        self.error = error-model.outputlfp
        self.learning_rate = learning_rate
        self.derivative = model.inputl.derivative, model.hiddenl.derivative, model.outputl.derivative
        self.weights = model.inputw, model.hiddenw
class modelArchitecture:
    def __init__(self,model):
        self.model = model
        print("Bias Value:", model.bias)
        print("Input Layer Nodes:", model.inputl.nodes)
        print("Hidden Layer Nodes:", model.hiddenl.nodes)
        print("Output Layer Nodes:", model.outputl.nodes)
        print("Activation Functions:", model.input_activation,model.hidden_activation, model.output_activation)
class saveModel:
    def __init__(self, model):
        self.model = model
        with open('model.pickle', 'wb') as f:
            pickle.dump(self.model,f)
class loadModel:
    def __init__(self,model):
        with open('model.pickle', 'rb') as f:
            model = pickle.load(f)
class deleteSavedModel:
    def __init__(self, model):
        if isinstance(model, str) and path.isfile(model+".pickle"):
            remove(model+".pickle")
        else:
            raise ValueError("File doesn't exist.")
