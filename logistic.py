# -*- coding: UTF-8 -*-

#================================================================
#   Copyright (C) 2019 OmniSci. All rights reserved.
#
#   Title：logistic.py
#   Author：Yong Bai
#   Time：2019-07-28 15:11:31
#   Description：
#
#================================================================

import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

dataset = np.loadtxt('watermelon.csv',delimiter=',')

X = dataset[:,1:3]
y = dataset[:,-1]

X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size=0.5,random_state = 0)


class MyLogReg:

    def likelihood_sub(self,x,y,beta):
        return (-y * np.dot(beta,x.T) + np.math.log(1 + np.math.exp(np.dot(beta,x.T))))

    def likelihood(self,X,y,beta):
        sum = 0
        m,_ = np.shape(X)

        for i in range(m):
            sum += self.likelihood_sub(X[i],y[i],beta)

        return sum

    def gradDscent(self,X,y):
        m, n = np.shape(X)

        max_times = 1000
        h = 0.1
        beta = np.random.rand(n)

        delta = np.random.rand(n)

        llh = 0
        llh_tmp = 0

        for i in range(max_times):
            beta_tmp = beta.copy()

            # 求偏导
            for j in range(n):
                beta_tmp[j] += delta[j]
                llh_tmp = self.likelihood(X,y,beta_tmp)

                delta[j] = -h*(llh_tmp - llh) / delta[j]

                beta_tmp[j] = beta[j]

            beta += delta
            llh = self.likelihood(X,y,beta)

            print (beta,llh)

        return beta

    def fit(self,X,y):
        self.beta = self.gradDscent(X,y)

    def predict(self,X):

        '''
        ln(y/1-y) 与0做比较,大于0表示正例,小于0表示反例
        '''

        y_pred = np.dot(self.beta,X.T)
        y_tmp = y_pred.copy()
        y_pred[y_tmp>=0] = 1.0
        y_pred[y_tmp<0] = 0

        return y_pred


#log_model = LogisticRegression()

log_model = MyLogReg()

log_model.fit(X_train,y_train)

y_pred = log_model.predict(X_test)

#from IPython import embed
#embed()

print (y_pred)

#print (metrics.confusion_matrix(y_test,y_pred))

print (metrics.classification_report(y_test,y_pred))








