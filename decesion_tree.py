# -*- coding: UTF-8 -*-

#================================================================
#   Copyright (C) 2019 OmniSci. All rights reserved.
#
#   Title：decesion_tree.py
#   Author：Yong Bai
#   Time：2019-07-30 14:40:51
#   Description：
#
#================================================================

import pandas as pd
from math import log
from sklearn.model_selection import train_test_split
from sklearn import metrics

def load_data():
    data = pd.read_csv('watermelon_3.csv')

    data = data.drop(columns = [u'编号'])

    dis_attr = [u'色泽',u'根蒂',u'敲声',u'纹理',u'脐部',u'触感']

    for attr in dis_attr:
        data[attr] = data[attr].astype('category').cat.codes

    data[u'好瓜'] = data[u'好瓜'].astype('category').cat.codes

    return train_test_split(data,test_size = 0.2)

class Node(object):
    def __init__(self):
        self.attr = None
        self.label = None
        self.child = {}


class Tree(object):
    def __init__(self,data):
        self.root = Node()
        self.label = u'好瓜'
        self.dis_attr = [u'色泽',u'根蒂',u'敲声',u'纹理',u'脐部',u'触感']
        self.conti_attr = [u'密度',u'含糖率']
        self.allData = data

    def sameAttr(self,data):
        data = data.drop(columns=[self.label])
        for idx,col in data.iteritems():
            if len(col.unique())>1:
                return False

        return True

    def mostLabel(self,data):
        counts = data[self.label].value_counts()
        max_num = max(counts)
        for label in data[self.label].unique():
            if counts[label]==max_num:
                return label

    def entropy(self,data):
        ent = 0
        counts = data[self.label].value_counts()
        n = data.shape[0]
        for count in counts:
            p = float(count)/float(n)
            ent -= p * log(p,2)

        return ent

    def InfoGain(self,attr,data,isConti):
        # continue attribute
        if isConti:
            values = sorted(data[attr].unique())
            maxV = -1
            maxThres = None
            for i in range(len(values)-1):
                thres = (values[i] + values[i+1]) / 2.
                nega = data[data[attr]<=thres]
                posi = data[data[attr]>thres]

                gain = -(nega.shape[0] * self.entropy(nega) + posi.shape[0] * self.entropy(posi)) / data.shape[0]

                if maxThres==None or gain>maxV:
                    maxThres = thres
                    maxV = gain

            return maxThres,maxV
        else:
            values = sorted(data[attr].unique())
            gain = 0
            for v in values:
                partition = data[data[attr]==v]
                gain -= partition.shape[0] * self.entropy(partition) / data.shape[0]

            return None,gain


    def OptAttr(self,data):
        infoGain = {}

        maxItem = None
        maxGain = 0

        for item in self.dis_attr:
            if item in data:
                infoGain[item] = self.InfoGain(item,data[[item,self.label]],False)
                if maxItem==None or infoGain[item][1]>maxGain:
                    maxItem = item
                    maxGain = infoGain[item][1]

        for item in self.conti_attr:
            if item in data:
                infoGain[item] = self.InfoGain(item,data[[item,self.label]],True)
                if maxItem==None or infoGain[item][1]>maxGain:
                    maxItem = item
                    maxGain = infoGain[item][1]

        return maxItem,infoGain[maxItem][0]

    def train(self,data):
        curr_node = Node()
        labels = data[self.label].unique()

        # only one class
        if len(labels)==1:
            curr_node.label = labels[0]
            return curr_node

        # no attr or all attr the same value
        if data.shape[1]<=1 or self.sameAttr(data):
            curr_node.label = self.mostLabel(data)
            return curr_node

        attr, div_value = self.OptAttr(data)


        curr_node.attr = attr
        # discrete attribute
        if div_value==None:
            values = self.allData[attr].unique()
            for v in values:
                # Dv is empty
                if not v in data[attr]:
                    curr_node.child[v] = Node()
                    curr_node.child[v].label = self.mostLabel(data)
                else:
                    data_tmp = data[data[attr]==v].copy()
                    data_tmp = data_tmp.drop(columns=[attr])
                    curr_node.child[v] = self.train(data_tmp)

        else:
            data_nega = data[data[attr]<=div_value]
            data_posi = data[data[attr]>div_value]
            if data_nega.shape[0]==0:
                curr_node.child['nega'] = Node()
                curr_node.child['nega'].label = self.mostLabel(data)
            else:
                curr_node.child['nega'] = self.train(data_nega)

            if data_posi.shape[0]==0:
                curr_node.child['posi'] = Node()
                curr_node.child['posi'].label = self.mostLabel(data)
            else:
                curr_node.child['posi'] = self.train(data_posi)
            curr_node.child['thres'] = div_value

        return curr_node

    def test(self,x,curr_node):
        if curr_node.label!=None:
            return curr_node.label

        attr = curr_node.attr
        if 'thres' in curr_node.child:
            if x[attr]<=curr_node.child['thres']:
                return self.test(x,curr_node.child['nega'])
            else:
                return self.test(x,curr_node.child['posi'])
        else:
            return self.test(x,curr_node.child[x[attr]])

    def predict(self,X):
        y_pred = []
        for i in range(X.shape[0]):
            y_pred.append(self.test(dict(X.iloc[i]),self.root))

        return y_pred




if __name__=='__main__':
    train,test = load_data()
    decision_model = Tree(train)
    decision_model.root = decision_model.train(train)
    #for i in range(train.shape[0]):
    #    print('result:',decision_model.test(dict(train.iloc[i]),decision_model.root))
    y_pred = decision_model.predict(test.drop(columns=[u'好瓜']))
    y_test = list(test[u'好瓜'])

    print (y_pred)

    #print (metrics.confusion_matrix(y_test,y_pred))

    print (metrics.classification_report(y_test,y_pred))

