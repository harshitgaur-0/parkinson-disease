#model 1
import pandas as pd
import numpy as np
df=pd.read_csv("parkinsons.data")
df.dropna()
dependent=df[["status"]]
independent=df.loc[:, df.columns != "status"]
independent=independent.loc[:, independent.columns != "name"]
from sklearn.preprocessing import MinMaxScaler
Scaler=MinMaxScaler((-1,1))
x=Scaler.fit_transform(independent,dependent)
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(independent,dependent,test_size=0.2)
from xgboost import XGBClassifier
model=XGBClassifier.fit(train_x,train_y)
pred_y=model.predict(test_x)
from sklearn.metrics import accuracy_score
print("accuracy of model=",accuracy_score(test_y,pred_y))
