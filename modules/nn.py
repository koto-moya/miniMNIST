from modules.engine import Value
import numpy as np
import random

class Neuron:
    def __init__(self, nin):
        self.w = np.array([Value(random.uniform(-1,1)) for _ in range(nin)])
        self.b = Value(random.uniform(-1,1)) #np.array([Value(random.uniform(-1,1))])

    def __call__(self, x):
        act =  sum((wi*xi for wi,xi in zip(self.w , x)), self.b)
        out = act.tanh()
        return out
    
    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, nout):
        self.layer = np.array([Neuron(nin) for _ in range(nout)])

    def __call__(self, x):
        preds = np.array([n(x) for n in self.layer])
        return preds
    
    def parameters(self):
         return [p for n in self.layer for p in n.parameters()]
    
class Model():
    def __init__(self, nin, outs: list):
        self.nin = nin
        self.outs = outs
        lmp = [nin] + outs
        self.layers = [Layer(lmp[i], lmp[i+1]) for i in range(len(outs))]

    def __repr__(self) -> str:
        return f"Model: \n Input Layer: {self.nin} \n layers: {self.outs}"
    
    def __call__(self, x):
        for lyr in self.layers:
            x = lyr(x)
        return x
    
    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]