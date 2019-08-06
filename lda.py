# -*- coding: UTF-8 -*-

#================================================================
#   Copyright (C) 2019 OmniSci. All rights reserved.
#
#   Title：lda.py
#   Author：Yong Bai
#   Time：2019-07-28 17:01:36
#   Description：
#
#================================================================

import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import metrics

dataset = np.loadtxt('watermelon.csv',delimiter=',')

X = dataset[:,1:3]
y = dataset[:,-1]

X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size=0.5,random_state = 0)


class MyLDA:
    def fit(self,X,y):
        self.u_0 = np.mean(X[y==0],axis=0)
        self.u_1 = np.mean(X[y==1],axis=0)

        m,n = np.shape(X)
        sw = np.zeros((n,n))
        for i in range(m):
            x_tmp = np.vstack(X[i])
            if y[i]==1:
                x_tmp -= np.vstack(self.u_1)
            else:
                x_tmp -= np.vstack(self.u_0)

            sw += np.dot(x_tmp,x_tmp.T)


        sw = np.mat(sw)
        U,sigma,V_trans = np.linalg.svd(sw)
        sw_inv = V_trans.T * np.linalg.inv(np.diag(sigma)) * U.T

        self.w = np.dot(sw_inv,np.vstack(self.u_0-self.u_1))

        return self.w

    def predict(self,X):
        map_u0 = np.dot(self.w.T,np.reshape(self.u_0,(2,1)))[0,0]
        map_u1 = np.dot(self.w.T,np.reshape(self.u_1,(2,1)))[0,0]

        threshold = (map_u0 + map_u1) / 2

        y_pred = np.array(np.dot(X_test,w).T)[0]
        y_tmp = y_pred.copy()

        y_pred[y_tmp>=threshold] = 0
        y_pred[y_tmp<threshold] = 1

        return y_pred






#lda_model = LinearDiscriminantAnalysis(solver='eigen',shrinkage=None)

lda_model = MyLDA()

w = lda_model.fit(X_train,y_train)


#print (w,np.shape(w))




y_pred = lda_model.predict(X_test)


print (y_pred)

#print (metrics.confusion_matrix(y_test,y_pred))

print (metrics.classification_report(y_test,y_pred))

