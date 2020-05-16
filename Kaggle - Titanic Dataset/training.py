import pandas as pd
import numpy as np
from sklearn import svm
from sklearn import tree
from sklearn import metrics
from sklearn import ensemble
from sklearn import neighbors
from sklearn import linear_model
from sklearn.model_selection import train_test_split as tts
from utilClasses import Utils, PreProcess, ValidateModels

utils = Utils()
preProcess = PreProcess()
validateModels = ValidateModels(10)

impCols = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
trainSet = pd.read_csv("Train_Subset.csv", sep=",", usecols=impCols)

labels = ['Survived']
trainLabels = trainSet[labels].astype('category')
trainAttrs = trainSet.drop(labels, axis=1)

X_train = preProcess.perform(trainAttrs)
Y_train = trainLabels.values.ravel()

nonlin_svm_model = svm.SVC(random_state=50, kernel="poly")
rf_model = ensemble.RandomForestClassifier(random_state=50, n_jobs=-1, max_depth=8)
gdb_model = ensemble.GradientBoostingClassifier(random_state=50, learning_rate=0.2)
voting_clf = ensemble.VotingClassifier(
            estimators=[('svc', nonlin_svm_model), ('rf', rf_model), ('gdb', gdb_model)],
            voting='hard'
        )
validateModels.perform(voting_clf, X_train, Y_train)