# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 12:21:20 2018

@author: baseline
"""

import pandas as pd
import lightgbm as lgb
from sklearn.metrics import f1_score,roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  
import numpy as np


'''
#看下缺失 直接拿0补缺失
miss_rate=train.apply(lambda x :sum(x.isnull())/11017)
'''

def feature_select(x_train,y_train):
    selected_feat_names=[]
    for i in range(10):                           #这里我们进行十次循环取交集
        tmp = set()
        rfc =RandomForestClassifier(n_estimators=100)
        rfc.fit(x_train, y_train)
        print("training finished")
    
        importances = rfc.feature_importances_
        indices = np.argsort(importances)[::-1]   # 降序排列
        for f in range(x_train.shape[1]):
            if f < 80 :                      #选出前80个重要的特征
                tmp.add(x_train.columns[indices[f]])
                print("%2d) %-*s %f" % (f + 1, 30, x_train.columns[indices[f]], importances[indices[f]]))
        
        selected_feat_names.append(tmp)          
        print(len(selected_feat_names), "features are selected")  
        #找出共同的前80特征
    list1=list(set(selected_feat_names[0]).intersection(*selected_feat_names[1:]))
    listchoose=list1
    return listchoose
    
def lgbCV(x_train, x_test,y_train,y_test):
    print('Training LGBM model...')
    lgb0 = lgb.LGBMClassifier(
        objective='binary',
        # metric='binary_error',
        num_leaves=35,
        max_depth=5,
        learning_rate=0.1,
        seed=417,
        colsample_bytree=0.8,
         min_child_samples=5,
        subsample=0.8,
        n_estimators=1000)
    lgb_model = lgb0.fit(x_train, y_train, eval_set=[(x_test, y_test)], early_stopping_rounds=200)
    best_iter = lgb_model.best_iteration_
    #predictors = [i for i in x_train.columns]
    #feat_imp = pd.Series(lgb_model.feature_importance(), predictors).sort_values(ascending=False)
    # print(feat_imp)
    #print(feat_imp.shape)
    # pred= lgb_model.predict(test[col])
    pred = lgb_model.predict(x_test)
    pred_train = lgb_model.predict(x_train)
    print('f1_score on train', f1_score(y_train, pred_train))
    print('f1_score on test', f1_score(y_test, pred))
    return best_iter

def sub(x_train, x_test,y_train,best_iter):
    print('Training LGBM model...')
    lgb0 = lgb.LGBMClassifier(
        objective='binary',
        # metric='binary_error',
        num_leaves=35,
        max_depth=5,
        learning_rate=0.1,
        seed=417,
        colsample_bytree=0.8,
         min_child_samples=5,
        subsample=0.8,
        n_estimators=best_iter)
    lgb_model = lgb0.fit(x_train, y_train)
    predict = lgb_model.predict(x_test)
    return predict
#拿0填补缺失值

if __name__ == "__main__":
    print('Read_data...')
    train=model_sample
    train_impute=train.fillna(0).drop('user_id',axis=1)
    print('train_test_split...')
    x=train_impute.drop('y',axis=1)
    y=train_impute['y']
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=417)
    feature=feature_select(X_train,y_train)
    X_train_select=X_train[feature]
    X_test_select=X_test[feature]
    best_iter = lgbCV(X_train_select,X_test_select, y_train, y_test)
    verify_sample_select=verify_sample[feature].fillna(0)
    predict=sub(X_train_select,verify_sample_select,y_train,best_iter)
    predict_result = pd.concat([verify_sample['user_id'],pd.Series(predict)],axis=1)
    
    
  
    
    
