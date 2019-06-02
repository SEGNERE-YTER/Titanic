# -*- coding: utf-8 -*-
"""
Created on Thu May 30 18:27:23 2019

@author: Marine
"""

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import csv

clf = RandomForestClassifier(n_estimators=6, max_depth=3, random_state=0)

del data_train["Pclass"]
del data_train["Sex"]
del data_train["Embarked"]

del data_test["Pclass"]
del data_test["Sex"]
del data_test["Embarked"]

y = data_train["Survived"]
del data_train["Survived"]
clf.fit(data_train, y)

y_pred=clf.predict(data_test)

#ENREGISTRER LES DONNEES
mon_fichier = open("second_submission_random_forest.csv", "w", newline='') 
mywriter = csv.writer(mon_fichier)

for k in range(len(y_pred)):
    print([data_test["PassengerId"][k], y_pred[k]])
    mywriter.writerow([data_test["PassengerId"][k], y_pred[k]])

mon_fichier.close()
