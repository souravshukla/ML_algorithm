import numpy as np
# import matplotlib.pyplot as plt
from logistics_reg import *

from sklearn import datasets
iris = datasets.load_iris()

x = np.array(iris['data'])
y = np.array((iris['target']==2).astype(int))
y = y.reshape((len(y),1))

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)

np.random.seed(42)
th= gra(x_train,y_train,100,0.1)

print("the theta value after 100 iteration is:\n",th)

y_preds,score = predict_and_score(x_test,y_test,th,0.5)

print("The accuracy score is:",score)
