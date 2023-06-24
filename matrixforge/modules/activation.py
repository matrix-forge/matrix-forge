from numpy import *


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#
# -------ACTIVATION FUNCTIONS--------#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#

def relu(x):
    ''' 
    Rectified Linear Unit

    Arguments: x - input 'matrix' or 'number' input 'matrix' or 'number'

    Formula: f(x) = max(0,x)
    '''
    return maximum(0, x)


def elu(x, alpha=1.0):
    '''
    Exponential Linear Unit

    Arguments: x - input 'matrix' or 'number' 

    Formula: f(x) = 
    '''
    return where(x > 0, x, alpha * (exp(x) - 1))


def selu(x, alpha=1.67326, scale=1.0507):
    '''
    Scaled Exponential Linear Unit

    Arguments: x - input 'matrix' or 'number' 

    Formula: f(x) = 
    '''
    return scale * where(x > 0, x, alpha * (exp(x) - 1))


def gelu(x):
    '''
    Gaussian Error Linear Unit

    Arguments: x - input 'matrix' or 'number' 

    Formula: f(x) = 
    '''
    return 0


def sigmoid(x):
    '''
    Sigmoid

    Arguments: x - input 'matrix' or 'number' 

    Formula: f(x) = 1/(1+e^(-x))
    '''
    return 1/(1+exp(-x))


def softplus(x):
    '''
    Softplus

    Arguments: x - input 'matrix' or 'number' 

    Formula: f(x) = log(1+e^x)
    '''
    return log(1+e**x)


def softmax(x):
    '''
    Softmax

    Arguments: x - input 'matrix' or 'number' 

    Formula: f(x) = 
    '''
    e_x = exp(x - max(x))
    return e_x / e_x.sum(axis=0)


def swish(x):
    '''
    Swish

    Arguments: x - input 'matrix' or 'number' 

    Formula: f(x) = x * 1/(1+e^(-x))
    '''
    return x * sigmoid(x)


def tanhh(x):
    '''
    Hyperbolic Tangent

    Arguments: x - input 'matrix' or 'number' 

    Formula: f(x) = 
    '''
    return tanh(x)


def hardtanh(x, min_val=-1, max_val=1):
    '''
    Hard Hyperbolic Tangent 

    Arguments: x - input 'matrix' or 'number' 

    Formula: f(x) = 
    '''
    return clip(x, min_val, max_val)


def linear(x):
    '''
    Linear Regression

    Arguments: x - input 'matrix' or 'number' 

    Formula: f(x) = x
    '''
    return x

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#
# -----------DERIVATIVES-------------#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#


def d_relu(x):
    '''
    Rectified Linear Unit Derivative

    Arguments: x - input 'matrix' or 'number' 

    Formula: f(x) = 1 * (x > 0)
    '''
    return 1. * (x > 0)


def d_elu(x, alpha=1.0):
    '''
    Exponential Linear Unit Derivative

    Arguments: x - input 'matrix' or 'number' 

    Formula: f(x) = 
    '''
    return alpha * exp(x) if x < 0 else 1


def d_selu(x, alpha=1.67326, scale=1.0507):
    '''
    Scaled Exponential Linear Unit Derivative

    Arguments: x - input 'matrix' or 'number' 

    Formula: f(x) = 
    '''
    return scale * (alpha * exp(x) if x < 0 else 1)


def d_gelu(x):
    '''
    Gaussian Error Linear Unit Derivative

    Arguments: x - input 'matrix' or 'number' 

    Formula: f(x) = 
    '''
    return 0


def d_sigmoid(x):
    '''
    Sigmoid Derivative

    Arguments: x - input 'matrix' or 'number' 

    Formula: f(x) = 1/(1+e^(-x)) * (1-x)
    '''
    a = 1/(1+e**(-x))
    return a*(1-a)


def d_softplus(x):
    '''
    Softplus Derivative

    Arguments: x - input 'matrix' or 'number' 

    Formula: f(x) = 1/(1+e^(-x))
    '''
    return 1/(1+e**(-x))


def d_softmax(x):
    '''
    Softmax Derivative

    Arguments: x - input 'matrix' or 'number' 

    Formula: f(x) = 
    '''
    soft = exp(x) / sum(exp(x), axis=0)
    return diag(soft) - outer(soft, soft)


def d_tanh(x):
    '''
    Hyperbolic Tangent Derivative

    Arguments: x - input 'matrix' or 'number' 

    Formula: f(x) = 
    '''
    return 1.-tanh(x)**2


def d_hardtanh(x):
    '''
    Hard Hyperbolic Tangent Derivative

    Arguments: x - input 'matrix' or 'number' 

    Formula: f(x) = 
    '''
    return where(x < -1, 0, where(x > 1, 0, 1))


def d_linear(x, y):
    '''
    Linear Regression Derivative

    Arguments: x - input 'matrix' or 'number' 

    Formula: f(x) = y
    '''
    return y


class Layer:
    def __init__(self, nodes, activation=''):
        self.nodes = nodes
        self.activation = activation
        self.derivative = None

        activation_functions = {
            'relu': (relu, d_relu),
            'elu': (elu, d_elu),
            'selu': (selu, d_selu),
            'gelu': (gelu, d_gelu),
            'sigmoid': (sigmoid, d_sigmoid),
            'softplus': (softplus, d_softplus),
            'softmax': (softmax, d_softmax),
            'tanh': (tanhh, d_tanh),
            'hardtanh': (hardtanh, d_hardtanh),
            'linear': (linear, d_linear)
        }

        if activation in activation_functions:
            self.activation, self.derivative = activation_functions[activation]
        else:
            raise ValueError("Unsupported activation function: ", activation)


class Input:
    def __init__(self, nodes):
        self.nodes = nodes
