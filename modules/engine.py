import numpy as np

class Value():
    def __init__(self, data, _children=(), _op = ""):
        self.data = np.float16(data) # raw numerical value
        self._prev = set(_children) # the connective tissue of the graph
        self._op = _op
        self._backward = lambda: None
        self.grad = 0.0    

    # __r funcs are fallback functions for python to use when it can't explicitly do the op you invoke
    # repr autmatically returns the string value when the object is called
    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad})"
    
    def exp(self):
        out = Value(np.exp(self.data), (self,))
        def _backward():
            self.grad += out.data * out.grad 
        out._backward = _backward
        return out
    
    def log(self):
        out =Value(np.log(self.data), (self,))
        def _backward():
            self.grad += (1/self.data) * out.grad 
        out._backward = _backward
        return out
    
    def tanh(self):
        t = ((np.exp(2*self.data))-1)/((np.exp(2*self.data))+1)
        out = Value(t, (self,), "tanh")
        def _backward():
            self.grad += (1-t**2) * out.grad
        out._backward = _backward
        return out
    
    def relu(self):
        out = Value(0 if self.data<0 else self.data, (self,), "ReLU")
        def _backward():
            self.grad += (out.data>0) * out.grad
        out._backward = _backward
        return out
    
    # Where other is assumed to be a value object as well? Yes
    def __add__(self, other):
        other  = other if isinstance(other, Value) else Value(other) # checking if the other number is a Value type, if not, wrap in Value
        out = Value(self.data + other.data, (self, other), "+")
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
            pass
        out._backward = _backward
        return out
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        out = self + (-other)
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other  = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        return self * other ** -1
    
    def __pow__(self, other):
        assert isinstance(other, (int,float))
        out = Value(self.data**other, (self,))
        def _backward():
            self.grad += (other*self.data**(other-1))*out.grad
        out._backward = _backward
        return out

    def __gt__(self, other):
        other  = other if isinstance(other, Value) else Value(other)
        return self.data > other.data

    def __lt__(self, other):
        other  = other if isinstance(other, Value) else Value(other)
        return self.data < other.data

    def __ge__(self, other):
        other  = other if isinstance(other, Value) else Value(other)
        return self.data >= other.data

    def __le__(self, other):
        other  = other if isinstance(other, Value) else Value(other)
        return self.data <= other.data
    
    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()