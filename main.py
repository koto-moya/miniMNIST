from modules.engine import Value
from modules.nn import Layer, Neuron, Model
from modules.DL import DataLoader
import numpy as np
import matplotlib.pyplot as plt

# implement a nn in nothing but numpy

# Init weights and bias

# set hyperparameters
    # for every epoch
        # make predictions

        # calculate loss 

        # backprop

        # Update the params


# Model will train on MNIST data (28X28) image of b&w pixel values

# Each image comes with a prediction so ([28*28],[0,9])

# Model should ingest image and make a prediction 0-9 calc loss then back prop

# model should have the layer structure Linear -> relu -> linear


class Trainer():
    def __init__(self, ds, model, lr, epochs):
        self.model = model
        self.lr = lr
        self.loss = 0.0
        self.epochs = epochs
        self.train_data = ds[round(len(ds)*.2):]
        self.valid_data = ds[:round(len(ds)*.2)]
        self.backward = np.vectorize(lambda x: x.backward()) 

    def train_loop(self):
        for _ in range(self.epochs):
            self.train()
            print(self.loss)

    def train(self):
        for x, y in self.train_data:
            p = self.model(x)
            self.loss = self.cross_entropy_loss(p, y)
            self.backward(self.loss)
            self.update_params

    def update_params(self):
        for p in self.model.parameters():
            p.data -= self.lr*p.grad
            p.grad = 0.0 

    def softmax(self, p):
        p = p-np.max(p)
        return np.exp(p)/np.sum(np.exp(p))
    
    def cross_entropy_loss(self, p, y):
        log_soft = np.log(self.softmax(p))
        one_hot = log_soft[range(len(log_soft)),y]
        cel = -np.sum(one_hot)
        return cel

if __name__ == "__main__":
    dl = DataLoader(); imgs = dl.imgs()#; model = Model(28*28, [30, 10]); train = Trainer(imgs, model, 0.01, 10)
    #train.train_loop()
    #plt.imshow(imgs[90][0].reshape(28,28))
    #print(imgs[1])


    