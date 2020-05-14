import pandas as pd
from utilClasses import Utils, Enhance, CleanData

utils = Utils()
enhance = Enhance()
cleanData = CleanData()

impCols = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
trainSet = pd.read_csv("Train_Subset.csv", sep=",", usecols=impCols)

labels = ['Survived']
trainLabels = trainSet[labels].astype('category')
trainAttrs = trainSet.drop(labels, axis=1)


#Exploration
print(trainAttrs.dtypes)
print(utils.getNAStats(trainAttrs))
print(trainAttrs.head(10))

#PreProcessing
trainAttrs = enhance.perform(trainAttrs)
trainAttrs = cleanData.perform(trainAttrs)

print(trainAttrs.head(10))