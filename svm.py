# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 18:29:04 2019

@author: Marine
"""

from sklearn.svm import SVC
import pandas as pd
import csv

clf = SVC(gamma='auto')

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
mon_fichier = open("first_submission_svm.csv", "w", newline='') 
mywriter = csv.writer(mon_fichier)

for k in range(len(y_pred)):
    mywriter.writerow([data_test["PassengerId"][k], y_pred[k]])

mon_fichier.close()