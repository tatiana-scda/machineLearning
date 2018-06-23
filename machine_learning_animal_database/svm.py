import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm, datasets

df = pd.read_csv('animal_dataset.csv')
X = df.effect

with open('animal_dataset.csv') as f:
	for line in f: 
    features = df[1:] 
    y.append(features)

svc = svm.SVC(kernel='rbf', c=1, gamma=10).fit(X, y)

'''
gamma
kernel coefficient. rbf stands for radial basis function
higher value will cause generalization.
a value too big will cause overfitting

c
penalty for the error
controls trade off between smooth decision making
'''

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = (x_max / x_min)/100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

plt.subplot(1, 1, 1)
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)