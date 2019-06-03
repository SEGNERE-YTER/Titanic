# -*- coding: utf-8 -*-
"""
Created on Thu May 30 17:04:43 2019

@author: Marine
"""

import numpy as np
import pandas as pd
import re

#lire données
data_train = pd.read_csv("train.csv")
data_test = pd.read_csv("test.csv")

#mise en forme des données#

#supprimer tickets
del data_train["Ticket"]
del data_test["Ticket"]
#supprimer cabine
del data_train["Cabin"]
del data_test["Cabin"]

#supprimer Name


data_train['Status'] = data_train['Name'].str.extract('\,(.*)\.', expand=True)
data_test['Status'] = data_test['Name'].str.extract('\,(.*)\.', expand=True)
del data_train["Name"]
del data_test["Name"]




#tableau disjonctif Pclass, sex, gate
pclass = pd.get_dummies(data_train["Pclass"])
sex = pd.get_dummies(data_train["Sex"])
embarked = pd.get_dummies(data_train["Embarked"])
status_train = pd.get_dummies(data_train["Status"])

del status_train[" Capt"]
del status_train[' Don']
del status_train[' Jonkheer']
del status_train[' Lady']
del status_train[' Major']
del status_train[' Mlle']
del status_train[' Mme']
del status_train[' Mrs. Martin (Elizabeth L']
del status_train[' Sir']
del status_train[' the Countess']
pclass.columns = ['class 1','class 2','class 3']
data_train = data_train.join(pclass)
data_train = data_train.join(status_train)
data_train = data_train.join(sex)

embarked.columns = ['embarked C','embarked Q','embarked S']
data_train = data_train.join(embarked)

pclass = pd.get_dummies(data_test["Pclass"])
sex = pd.get_dummies(data_test["Sex"])
embarked = pd.get_dummies(data_test["Embarked"])
status_test = pd.get_dummies(data_test["Status"])
del status_test[' Dona']



pclass.columns = ['class 1','class 2','class 3']
data_test = data_test.join(pclass)
data_test = data_test.join(status_test)
data_test = data_test.join(sex)

embarked.columns = ['embarked C','embarked Q','embarked S']
data_test = data_test.join(embarked)

#centrer et réduire
data_train_f = data_train[data_train["Sex"] == 'female']
data_train_m = data_train[data_train["Sex"] == 'male']

data_test_f = data_test[data_test["Sex"] == 'female']
data_test_m = data_test[data_test["Sex"] == 'male']



#moyenne_age = np.mean(data_train['Age'])
#ecart_type_age = np.std(data_train['Age'])
#data_train['Age'] = (data_train['Age'] - moyenne_age)/ecart_type_age
#data_test['Age'] = (data_test['Age'] - moyenne_age)/ecart_type_age

moyenne_age_f = np.mean(data_train_f['Age'])
ecart_type_age_f = np.std(data_train_m['Age'])
data_train_f['Age'] = (data_train_f['Age'] - moyenne_age_f)/ecart_type_age_f
data_test_f['Age'] = (data_test_f['Age'] - moyenne_age_f)/ecart_type_age_f

moyenne_age_m = np.mean(data_train[data_train["Sex"] == 'male']['Age'])
ecart_type_age_m = np.std(data_train[data_train["Sex"]== 'male']['Age'])
data_train_m["Age"] = (data_train_m["Age"] - moyenne_age_m)/ecart_type_age_m
data_test_m["Age"] = (data_test_m["Age"] - moyenne_age_m)/ecart_type_age_m

#data_test["Age"][np.isnan(data_test["Age"])] = np.mean(data_test['Age'])
data_test_f["Age"][np.isnan(data_test_f["Age"])] = np.mean(data_test_f["Age"])
data_test_m['Age'][np.isnan(data_test_m["Age"])] = np.mean(data_test_m["Age"])

data_train = pd.concat([data_train_f, data_train_m])
data_test = pd.concat([data_test_f, data_test_m])

moyenne_SibSp = np.mean(data_train['SibSp'])
ecart_type_SibSp = np.std(data_train['SibSp'])
data_train['SibSp'] = (data_train['SibSp'] - moyenne_SibSp)/ecart_type_SibSp
data_test['SibSp'] = (data_test['SibSp'] - moyenne_SibSp)/ecart_type_SibSp

moyenne_Parch = np.mean(data_train['Parch'])
ecart_type_Parch = np.std(data_train['Parch'])
data_train['Parch'] = (data_train['Parch'] - moyenne_Parch)/ecart_type_Parch
data_test['Parch'] = (data_test['Parch'] - moyenne_Parch)/ecart_type_Parch

moyenne_Fare = np.mean(data_train['Fare'])
ecart_type_Fare = np.std(data_train['Fare'])
data_train['Fare'] = (data_train['Fare'] - moyenne_Fare)/ecart_type_Fare
data_test['Fare'] = (data_test['Fare'] - moyenne_Fare)/ecart_type_Fare

#remplacer valeur "NaN" dans Age
#col_mask=data_test.isnull().any(axis=0) 
data_train["Age"][np.isnan(data_train["Age"])] = 0

data_test["Fare"][np.isnan(data_test["Fare"])] = np.mean(data_test['Fare'])

data_train = data_train.sort_values(by = 'PassengerId')
data_test = data_test.sort_values(by = 'PassengerId')

del data_train["Status"]
del data_test["Status"]


