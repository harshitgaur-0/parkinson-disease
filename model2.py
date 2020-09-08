import pandas as pd
import numpy as np
df=pd.read_csv("parkinsons.data")
df.dropna()
dependent=df[["status"]]
independent=df.loc[:, df.columns != "status"]
independent=independent.loc[:, independent.columns != "name"]
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
norm_x=preprocessing.StandardScaler().fit(independent).transform(independent)
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(norm_x,dependent,test_size=0.2)
neigh = KNeighborsClassifier(n_neighbors = 2).fit(train_x,train_y.values.ravel())
y_pred=neigh.predict(test_x)
from sklearn.metrics import accuracy_score
print("accuracy of model=",accuracy_score(test_y,y_pred))
