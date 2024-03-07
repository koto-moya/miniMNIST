from modules.engine import Value
import numpy as np
import random

class Neuron:
    def __init__(self, nin):
        self.w = np.array([Value(random.uniform(-1,1)) for _ in range(nin)])
        self.b = np.array([Value(random.uniform(-1,1))])

    # call returns when the insatiated class is used like: n = Neuron(2) -> n(3) -> out of __call__    
    def __call__(self, x): # x should be a numpy array
        return sum(self.w * x, self.b) 
    
    def parameters(self):
        return self.w + self.b

class Layer:
    def __init__(self, nin, nout):
        self.layer = np.array([Neuron(nin) for _ in range(nout)])

    def __call__(self, x, act = np.tanh):
        act = np.array([n(x) for n in self.layer])
        relu = np.vectorize(lambda x: x.relu())
        return relu(act)
    
    def parameters(self):
         return [p for n in self.layer for p in n.parameters()]
    
class Model():
    def __init__(self, nin, outs: list):
        lmp = [nin] + outs
        self.layers = [Layer(lmp[i], lmp[i+1]) for i in range(len(outs))]
    
    def __call__(self, x):
        for lyr in self.layers:
            x = lyr(x)
        return x
    
    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]