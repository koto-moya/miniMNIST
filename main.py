from modules.DL import DataLoader
from modules.nn import Model

import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import trange


if __name__ == "__main__":
    dl = DataLoader()
    epochs = 200
    model = Model(28*28, [10,10])

    def logsoft(p):
        p = p-np.max(p)
        return np.log(np.exp(p)/np.sum(np.exp(p)))
    
    def cross_entropy_loss(p, y):
        log_soft = logsoft(p)
        one_hot = log_soft[y]
        cel = -np.sum(one_hot)
        return cel
    
    
    lr = 0.01
    losses = []
    accuracies = []
    batch_losses = []
    batch_accuracies = []
    start_time = time.monotonic()
    for i in  range(epochs):
        xy_zip = dl.imgs()
        test = dl.imgs(ds_type = "test")
        for x, y in xy_zip:
            p = model(x)
            loss = cross_entropy_loss(p, y)
            loss.backward()
            batch_losses.append(loss.data)
            for p in model.parameters():
                p.data -= lr*p.grad
                p.grad = 0.0
        for x,y in test:
            p = model(x)
            batch_accuracies.append(np.argmax(p)==y)
        accuracies.append(np.mean(batch_accuracies))
        batch_accuracies = []
        losses.append(np.mean(batch_losses))
        print(f"loss: {np.mean(batch_losses): .2f}    |   testing accuracy: {np.mean(accuracies)*100: .2f}%    |   epoch: {i}", end="\r", flush=True)
        batch_losses = []
    end_time = time.monotonic()
    print(f"time taken: {(end_time-start_time)/60: .2f} mins", end="\r", flush=True)
    plt.plot(losses)
    plt.show()



    