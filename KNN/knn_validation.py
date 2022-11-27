from sklearn import datasets
from sklearn.model_selection import train_test_split
import knn
import numpy as np

iris = datasets.load_iris()
Data_train, Data_test, labels_train, labels_test = train_test_split(iris.data, iris.target, test_size=.1, random_state=3214)

Alg = knn.knn()
Alg.train(Data_train, labels_train)

learned_labels = Alg.learn(Data_test)
error = np.sum( (labels_test != learned_labels) ) / len(labels_test)

print(f'Error in classification is {100*error:.2f}%')
