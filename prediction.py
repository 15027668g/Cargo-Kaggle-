# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 21:50:21 2017

@author: samsung
"""

import pandas as pd

from sklearn.feature_selection import chi2
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest

#read test data
testdata = pd.read_csv("test.csv", sep = ",",index_col = 0, header=0)
#read train data
traindata = pd.read_csv("train.csv", sep = ",",index_col = 0, header=0)
# extract the target out of traindata
traindata_label = pd.DataFrame(traindata["target"]) 
traindata= traindata.drop("target",1)

# basic statistic
traindata_describe = traindata.describe()

'''
#Chi-Square feature selection
def SelectKBest_selector(data,label):
    columns = data.columns
    selector = SelectKBest(chi2, k=57)
    selector.fit(data,label)
    labels = [columns[x] for x in selector.get_support(indices=True)]
    return pd.DataFrame(selector.fit_transform(data,label), columns=labels), selector.scores_

#Calculate Basic statistic value (Chi2)
traindata_sel_chi2, chi2_score= SelectKBest_selector(traindata, traindata_label)
traindata_summary.insert(loc=2,column="chi2",value=chi2_score)
#traindata_summary = data_norm_summary.round(3)
'''


