import numpy as np
import collections as cl

class knn:
    def __init__(self, k=3):
        self.k = k
    
    def train(self, data, labels):
        self.train_data = data
        self.train_labels = labels
    
    def learn(self, X):
        pred = [self._learn(x) for x in X]
        return np.array(pred)

    def _learn(self, x):
        dist = [np.linalg.norm(x-z, 2) for z in self.train_data]
        ind = np.argsort(dist)
        ind = ind[:self.k]
        label_near = self.train_labels[ind]
        return cl.Counter(label_near).most_common(1)[0][0]