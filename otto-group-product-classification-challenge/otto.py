#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 11:16:26 2019

@author: oguz
"""

import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn import svm
clf = svm.SVC(gamma='scale')
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
from sklearn.preprocessing import MinMaxScaler 
data=pd.read_csv("/home/oguz/Desktop/otto/train.csv")
test=pd.read_csv("/home/oguz/Desktop/otto/test.csv")

data=data.drop(['id'], axis=1) 
test=test.drop(['id'],axis=1)
data.iloc[:,-1]=le.fit_transform(data.iloc[:,-1]) #label encoder ile kategorik olan değerler nümerik değere dönüştürüldü 


#data.info() #eksik değer var mı kontrolu ? 

#print(data.describe())

scaler = MinMaxScaler()  #normalize edildi 0-1 arasında 
data.iloc[:,0:-1]=scaler.fit_transform(data.iloc[:,0:-1])
test.iloc[:,:]=scaler.fit_transform(test.iloc[:,:])


col=data.iloc[:,0:-1]
label=pd.DataFrame(data.iloc[:,-1])
features = data[list(col)].values
#p-value değerleri yüksek olanlar çıkarıldı. 


col=col.drop(['feat_5','feat_7', 'feat_9','feat_14','feat_23','feat_30','feat_31','feat_33','feat_40','feat_43','feat_49',
              'feat_55','feat_58','feat_61','feat_64','feat_72','feat_74','feat_78','feat_80','feat_87','feat_27','feat_90','feat_93'], axis=1)

test=test.drop(['feat_5','feat_7', 'feat_9','feat_14','feat_23','feat_30','feat_31','feat_33','feat_40','feat_43',
                'feat_49','feat_55','feat_58','feat_61','feat_64','feat_72','feat_74','feat_78','feat_80',
                'feat_87','feat_27','feat_90','feat_93'], axis=1)

results = sm.OLS(label, col).fit()
print(results.summary())
print("***********************************************************")

train_pca=PCA(n_components=40)
x_pca = train_pca.fit_transform(col)
x_pca=pd.DataFrame(x_pca)


features = pd.DataFrame(col.sum()[0:-1])
features.columns = ['values']

print("Maximum value: ", col[col.columns[0:-1]].max().max())
print("Minimum value: ", col[col.columns[0:-1]].min().min())
#
features.sort_values('values', ascending=True).plot(kind='barh', figsize=(10,20))

# PCA İle 40 değişkene indirgendi.Çok değişkenli verimizi daha az değişkenle temsil edebiliriz.Olabilecek 
#bilgi kaybını göz ardı ediyoruz.

pca_test=PCA(n_components=40)
test_pca = pca_test.fit_transform(test)
test_pca=pd.DataFrame(test_pca)

#KERAS
model = Sequential()
model.add(Dense(20, input_dim=40, init='uniform', activation='relu'))
model.add(Dense(12, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='sigmoid'))
model.add(Dense(1, init='uniform', activation='sigmoid')) 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_pca,label,epochs=50)

y_pred=model.predict(test_pca)

#SVM
clf.fit(x_pca, label)
prediction=clf.predict(test_pca) 



