import pandas as pd
from utilClasses import Split

impCols = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
trainSet = pd.read_csv("Train_Subset.csv", sep=",", usecols=impCols)

# labels = ['Survived']
# split = Split(0.2, labels, True, ['Pclass', 'Sex'] + labels)
# trainDF, testDF, Y_train, Y_test = split.perform(trainSet)
# print(trainDF)