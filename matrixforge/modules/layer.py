from numpy import *

class Model():
    def __init__(self, nodes):
        self.nodes = len(nodes)
        self.inputl = nodes[0]
        self.outputl = nodes[-1]
        if len(nodes) == 2:
            print('NWM')
        elif len(nodes) == 1:
            raise ValueError(
                "Model must contain at least 2 layers, Input and Output")
        else:
            K = len(nodes)-2
            middle = len(nodes) // 2
            self.hiddenl = [nodes[i].nodes for i in range(
                middle - K//2, middle + (K//2) + 1) if i != 0 and i != len(nodes) - 1][0]


def sgd(weights, gradient, learning_rate):
    for i in range(len(weights)):
        weights[i] = weights[i] - learning_rate * gradient[i]
    return weights


class modelCompute():
    def __init__(self, model, bias):
        self.model = model
        self.bias = bias
        if self.bias == None:
            self.bias = 0

        self.inputl = random.randn(model.inputl.nodes)
        self.w1 = random.randn(model.inputl.nodes, model.hiddenl)
        self.ac = model.outputl.activation
        self.hiddenl = dot(self.inputl, self.w1)+self.bias
