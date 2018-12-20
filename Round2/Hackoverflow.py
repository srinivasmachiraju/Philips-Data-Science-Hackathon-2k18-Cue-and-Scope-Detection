#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

dataset=pd.read_csv('train.csv')
X = dataset.iloc[:, [0,2,4,6,7,8,9,10,11,12,13,14,15,16]].values

Y=dataset.iloc[:, 20].values

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)



yin=[]
for y in Y:
    if y==0:
      yin.append([1,0,0,0])
    elif y==1:
      yin.append([0,1,0,0])
    elif y==2:
      yin.append([0,0,1,0])
    else:
      yin.append([0,0,0,1])
      
yin=np.array(yin)

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json

#initializing the ANN
classifier=Sequential()

#adding input and first hidden layer
classifier.add(Dense(output_dim=10,init='uniform',activation='relu',input_dim=14))

#adding the second hidden layer
classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))

#adding the output layer
classifier.add(Dense(output_dim=4,init='uniform',activation='sigmoid'))

#compiling the ANN
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Fitting classifier to the Training set

classifier.fit(X,yin,batch_size=2,epochs=120)



dataset1=pd.read_csv('test.csv')
X_final = dataset1.iloc[:, [1,3,5,7,8,9,10,11,12,13,14,15,16,17]].values

#Y=dataset.iloc[:, 20:].values

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_tes = sc_X.fit_transform(X_final)
yf=classifier.predict(X_tes)

y_final=[]
for y in yf:
    a=max(y)
    if y[0]==a:
            y_final.append(0)
    elif y[1]==a:
            y_final.append(1)
    elif y[2]==a:
             y_final.append(2)
    else:
             y_final.append(3)
         


y_final=np.array(y_final)
a = np.asarray(y_final)
np.savetxt("sub.csv", np.dstack((np.arange(1, a.size+1),a))[0],"%d,%d",header="id,price_range")