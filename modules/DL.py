import random
import pandas as pd
import numpy as np

class DataLoader():
    def __init__(self, batch_size = 60):
        self.train = pd.read_csv("data/mnist_train.csv")
        self.test = pd.read_csv("data/mnist_test.csv")
        self.bs = batch_size

    def imgs(self):
        imgs = np.array(self.train.iloc[:64, 1:])/255
        labels = np.array(self.train.iloc[:64, :1])
        return imgs, labels
        
        #random.shuffle(labeled_imgs)
        #print(labeled_imgs)
        #batched = [labeled_imgs[:i*self.bs] for i in range(round(len(labeled_imgs)/self.bs)) if i*self.bs < len(labeled_imgs)]
        #print(batched)
        #print(f"# of image/label pairs: {len(labeled_imgs)}")
        #return labeled_imgs