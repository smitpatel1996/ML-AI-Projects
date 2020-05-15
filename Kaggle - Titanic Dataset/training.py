import pandas as pd
from utilClasses import Utils, PreProcess

utils = Utils()
preProcess = PreProcess()

impCols = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
trainSet = pd.read_csv("Train_Subset.csv", sep=",", usecols=impCols)

labels = ['Survived']
trainLabels = trainSet[labels].astype('category')
trainAttrs = trainSet.drop(labels, axis=1)

#Exploration
print(trainLabels['Survived'].value_counts())
print(trainAttrs.dtypes)
print(trainAttrs.head())
print(utils.getNAStats(trainAttrs))

#PreProcessing
X_train = preProcess.perform(trainAttrs)
Y_train = trainLabels.values.ravel()