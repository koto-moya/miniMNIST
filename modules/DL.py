import pandas as pd
import numpy as np

class DataLoader():
    def __init__(self, batch_size = 64, shuffle=True):
        self.train = pd.read_csv("data/mnist_train.csv",nrows=10000)
        self.test = pd.read_csv("data/mnist_test.csv", nrows=1000)
        self.bs = batch_size
        self.shuffle = shuffle

    def batch_gen(self, data_type='train'):
        if data_type == 'train':
            data = self.train
        elif data_type == 'test':
            data = self.test
        else:
            raise ValueError("data_type must be 'train' or 'test'")
        if self.shuffle:
            data = data.sample(frac=1).reset_index(drop=True)
        total_size = len(data)
        for start_idx in range(0, total_size, self.bs):
            end_idx = min(start_idx + self.bs, total_size)
            imgs = np.array(data.iloc[start_idx:end_idx, 1:]) / 255.0
            labels = np.array(data.iloc[start_idx:end_idx, 0])
            yield imgs, labels

    def imgs(self, ds_type = "train"):
        if ds_type == "train":
            data = self.train.sample(frac=1, random_state=42).reset_index(drop=True)
        if ds_type == "test":
            data = self.test.sample(frac=1, random_state=42).reset_index(drop=True)
        imgs = np.array(data.iloc[:self.bs, 1:])/255
        labels = np.array(data.iloc[:self.bs, :1])
        return list(zip(imgs, labels))
        