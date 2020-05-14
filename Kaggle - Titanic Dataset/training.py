import pandas as pd
from utilClasses import Utils, Enhance, CleanData, CategoryConvert, ScaleFeature

utils = Utils()
enhance = Enhance()
cleanData = CleanData()
categoryConvert = CategoryConvert()
scaleFeature = ScaleFeature()

impCols = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
trainSet = pd.read_csv("Train_Subset.csv", sep=",", usecols=impCols)

labels = ['Survived']
trainLabels = trainSet[labels].astype('category')
trainAttrs = trainSet.drop(labels, axis=1)
samp = (trainAttrs.iloc[5:6,:])

#Exploration
# print(trainAttrs.dtypes)
# print(utils.getNAStats(trainAttrs))
# print(trainAttrs.head(10))

#PreProcessing
trainAttrs = enhance.perform(trainAttrs)
trainAttrs = cleanData.perform(trainAttrs)
trainAttrs = categoryConvert.perform(trainAttrs)
trainAttrs = scaleFeature.perform(trainAttrs)
# print(trainAttrs.head(10))
# print(utils.getNAStats(trainAttrs))
# print(trainAttrs.dtypes)

samp = enhance.perform(samp)
samp = cleanData.perform(samp, training=False)
samp = categoryConvert.perform(samp, training=False)
samp = scaleFeature.perform(samp, training=False)
