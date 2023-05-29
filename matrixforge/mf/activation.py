from numpy import *

def relu(x):
    return maximum(0, x)
def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))
def selu(x, alpha=1.67326, scale=1.0507):
    return scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))
def sigmoid(x):
    return 1/(1+exp(-x))
def softplus(x):
    return log(1+e**x)
def softmax(x):
    e_x = exp(x - max(x))
    return e_x / e_x.sum(axis=0)
def swish(x):
    return x * sigmoid(x)
def tanhh(x):
    return tanh(x)
def hardtanh(x, min_val=-1, max_val=1):
    return np.clip(x, min_val, max_val)
def d_relu(x):
    return 1. * (x > 0)
def d_elu(x, alpha=1.0):
    return alpha * np.exp(x) if x < 0 else 1
def d_selu(x, alpha=1.67326, scale=1.0507):
    return scale * (alpha * np.exp(x) if x < 0 else 1)
def d_sigmoid(x):
    a = 1/(1+e**(-x))
    return a*(1-a)
def d_softplus(x):
    return 1/(1+e**(-x))
def d_softmax(x):
    soft = exp(x) / sum(exp(x), axis=0)
    return diag(soft) - outer(soft, soft)
def d_tanh(x):
    return 1.-tanh(x)**2
def d_hardtanh(x):
    return np.where(x < -1, 0, np.where(x > 1, 0, 1))


class Layer:
    def __init__(self, nodes, activation=''):
        self.nodes = nodes
        self.activation = activation
        self.derivative = None
        
        activation_functions = {
            'relu': (relu, d_relu),
            'elu': (elu, d_elu),
            'selu': (selu, d_selu),
            'sigmoid': (sigmoid, d_sigmoid),
            'softplus': (softplus, d_softplus),
            'softmax': (softmax, d_softmax),
            'tanh': (tanhh, d_tanh),
            'hardtanh': (hardtanh, d_hardtanh)
        }
        
        if activation in activation_functions:
            self.activation, self.derivative = activation_functions[activation]
        else:
            raise ValueError("Unsupported activation function")