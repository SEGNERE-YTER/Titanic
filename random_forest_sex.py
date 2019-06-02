# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 15:33:22 2019

@author: Marine
"""

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import csv

data_train_f = data_train[data_train["Sex"] == 'female']
data_train_m = data_train[data_train["Sex"] == 'male']

data_test_f = data_test[data_test["Sex"] == 'female']
data_test_m = data_test[data_test["Sex"] == 'male']

clf = RandomForestClassifier(bootstrap= True , min_samples_leaf= 3, n_estimators = 500 ,
                               min_samples_split = 10, max_features = "sqrt", max_depth= 6)

del data_train_f["Pclass"]
del data_train_f["Sex"]
del data_train_f["Embarked"]

del data_train_m["Pclass"]
del data_train_m["Sex"]
del data_train_m["Embarked"]

del data_test_f["Pclass"]
del data_test_f["Sex"]
del data_test_f["Embarked"]

del data_test_m["Pclass"]
del data_test_m["Sex"]
del data_test_m["Embarked"]

y_f = data_train_f["Survived"]
del data_train_f["Survived"]
clf.fit(data_train_f, y_f)

y_pred_f = clf.predict(data_test_f)
data_test_f['Survived'] = y_pred_f

y_m = data_train_m["Survived"]
del data_train_m["Survived"]
clf.fit(data_train_m, y_m)

y_pred_m = clf.predict(data_test_m)
data_test_m['Survived'] = y_pred_m

data_test = pd.concat([data_test_f, data_test_m])
data_test = data_test.sort_values(by = 'PassengerId')


#ENREGISTRER LES DONNEES
mon_fichier = open("third_submission_random_forest.csv", "w", newline='') 
mywriter = csv.writer(mon_fichier)

for k in range(len(y_pred)):
    print([data_test["PassengerId"][k], y_pred[k]])
    mywriter.writerow([data_test["PassengerId"][k], y_pred[k]])

mon_fichier.close()

