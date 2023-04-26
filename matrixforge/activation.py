from numpy import *

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
