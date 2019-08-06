# -*- coding: UTF-8 -*-

#================================================================
#   Copyright (C) 2019 OmniSci. All rights reserved.
#
#   Title：naive_bayes.py
#   Author：Yong Bai
#   Time：2019-08-06 14:37:20
#   Description：
#
#================================================================

import pandas as pd
from math import log,pi,exp,sqrt
from sklearn.model_selection import train_test_split
from sklearn import metrics


class NBC(object):
    def __init__(self):
        self.dis_attr = [u'色泽',u'根蒂',u'敲声',u'纹理',u'脐部',u'触感']
        self.conti_attr = [u'密度',u'含糖率']

    def load_data(self):
        self.data = pd.read_csv('watermelon_3.csv')
        self.cates = {}
        self.data = self.data.drop(columns = [u'编号'])

        for attr in self.dis_attr:
            category = self.data[attr].astype('category').cat
            self.data[attr] = category.codes
            self.cates[attr] = category.categories.tolist()

        category = self.data[u'好瓜'] .astype('category').cat
        self.data[u'好瓜'] = category.codes
        self.cates[u'好瓜'] = category.categories.tolist()

    def log_frac(self,a,b):
        return log(float(a)/float(b))

    def gauss(self,mean,std,value):
        return exp(-((value-mean)**2)/(2*std**2)) / (sqrt(2*pi)*std)

    def get_cate(self,attr,value):
        return self.cates[attr].index(value)

    def predict(self,X):
        group = self.data.groupby('好瓜')
        self.good_group = group.get_group(self.cates[u'好瓜'].index(u'是'))
        self.bad_group = group.get_group(self.cates[u'好瓜'].index(u'否'))
        n,_ = self.data.shape
        n_good,_ = self.good_group.shape
        n_bad,_ = self.bad_group.shape
        p_good = self.log_frac(n_good+1,n+2)
        p_bad = self.log_frac(n_bad+1,n+2)
        print ('先验',exp(p_good))
        for attr in self.dis_attr:
            n_good_attr = self.good_group[attr].value_counts()[self.cates[attr].index(X[attr])]
            n_bad_attr = self.bad_group[attr].value_counts()[self.cates[attr].index(X[attr])]

            p_good_delta = self.log_frac(n_good_attr+1,n_good+len(self.cates[attr]))
            p_bad_delta = self.log_frac(n_bad_attr+1,n_bad+len(self.cates[attr]))

            print (attr,exp(p_good_delta))
            p_good += p_good_delta
            p_bad += p_bad_delta

        group_mean = group.mean()
        group_std = group.std()
        for attr in self.conti_attr:
            mean_good = group_mean[attr][self.cates[u'好瓜'].index(u'是')]
            mean_bad = group_mean[attr][self.cates[u'好瓜'].index(u'否')]
            std_good = group_std[attr][self.cates[u'好瓜'].index(u'是')]
            std_bad = group_std[attr][self.cates[u'好瓜'].index(u'否')]

            p_good_delta = self.gauss(mean_good,std_good,X[attr])
            p_bad_delta = self.gauss(mean_bad,std_bad,X[attr])

            print(attr,p_good_delta)

            p_good += p_good_delta
            p_bad += p_bad_delta


        print (exp(p_good),exp(p_bad))
        if p_good > p_bad:
            return 'yes'
        else:
            return 'no'


if __name__=='__main__':
    #from IPython import embed
    #embed()

    X = {
            u'色泽':u'青绿',
            u'根蒂':u'蜷缩',
            u'敲声':u'浊响',
            u'纹理':u'清晰',
            u'脐部':u'凹陷',
            u'触感':u'硬滑',
            u'密度':0.697,
            u'含糖率':0.460
            }
    nbc_model = NBC()
    nbc_model.load_data()
    res = nbc_model.predict(X)
    print (res)

