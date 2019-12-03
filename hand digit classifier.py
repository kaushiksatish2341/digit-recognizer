import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier as clf

data=pd.read_csv('train.csv').as_matrix()

#data=data.fillna(method='ffill')

#label[label==' '] = np.median(label)

xtrain=data[0:30000,1:]
ytrain=data[0:30000,0]

clf.fit(xtrain,ytrain)

xtest=data[30000:,1:]
ytest=data[30000:,0]

a=xtest[1]
a.shape=(28,28)
plt.imshow(255-a,cmap='gray')
print(clf.predict([xtest[1]]))
plt.show()
