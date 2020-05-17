import random
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn import tree
from sklearn import metrics
from sklearn import ensemble
from sklearn import neighbors
from sklearn import linear_model
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFECV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score as cvs
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import cross_val_predict as cvp

class Utils:
    def getNAStats(self, dataFrame):
        print(dataFrame.isna().sum())
    
    def getCorrToLabels(self, attrs, labels):
        dataFrame = attrs
        dataFrame['Survived'] = labels
        corrMatrix = dataFrame.corr()
        print(corrMatrix['Survived'].sort_values(ascending=True))

class Split():
    def __init__(self, test_size, labels, stratify=False, stratifyBy=None):
        self.test_size = test_size
        self.labels = labels
        self.stratify = stratify
        self.stratifyBy = stratifyBy
        
    def __get_tts_from_df(self, dataFrame, test_size, attrs, labels, stratifyBy=None):
        if(self.stratify):
            train, test = tts(dataFrame, test_size=test_size, stratify=stratifyBy, random_state=50)
        else:
            train, test = tts(dataFrame, test_size=test_size, random_state=50)
        Y_train = (train[labels]).values.ravel()
        Y_test = (test[labels]).values.ravel()
        return (train[attrs], test[attrs], Y_train, Y_test)
    
    def perform(self, dataFrame):
        attrs = dataFrame.drop(self.labels, axis=1).columns.values.tolist()
        if(self.stratify):
            return self.__get_tts_from_df(dataFrame, self.test_size, attrs, self.labels, dataFrame[self.stratifyBy])
        else:
            return self.__get_tts_from_df(dataFrame, self.test_size, attrs, self.labels)

class Enhance():
    def __getSalutation(self, name):
        salut = (((name.split(", "))[1]).split(" "))[0]
        if(salut == "the"):
            salut = "Countess."
        return salut
    
    def __getCategory(self, desDF, colName, val):
        if(val <= desDF[colName].loc['25%']): return 'vl'
        elif(val < desDF[colName].loc['50%']): return 'l'
        elif(val == desDF[colName].loc['50%']): return random.choice(['l','h'])
        elif(val <= desDF[colName].loc['75%']): return 'h'
        else: return 'vh'
    
    def __getFamSize(self, val):
        if(val <= 3): return 'small'
        elif(val <= 6): return 'medium'
        else: return 'large'
    
    def setDtypes(self, dataFrame, dtypeDict):
        dataFrame = dataFrame.copy()
        for i in dtypeDict:
            if(i == 'string'):
                dataFrame[dtypeDict[i]] = dataFrame[dtypeDict[i]].astype(str)
            else:
                dataFrame[dtypeDict[i]] = dataFrame[dtypeDict[i]].astype(i)
        return dataFrame
    
    def perform(self, dataFrame, training=True):
        dataFrame['Name'] = dataFrame['Name'].apply(lambda x: self.__getSalutation(x))
        dataFrame['FamSize'] = dataFrame['SibSp'] + dataFrame['Parch'] + 1
        dataFrame = dataFrame.drop(['SibSp', 'Parch'], axis=1)
        if(training):
            self.desDF = dataFrame[['Age', 'Fare', 'FamSize']].describe()
        dataFrame['Age'] = dataFrame['Age'].apply(lambda x: self.__getCategory(self.desDF, 'Age', x))
        dataFrame['Fare'] = dataFrame['Fare'].apply(lambda x: self.__getCategory(self.desDF, 'Fare', x))
        dataFrame['FamSize'] = dataFrame['FamSize'].apply(lambda x: self.__getFamSize(x))
        dataFrame = self.setDtypes(dataFrame, {'category':['Name', 'FamSize', 'Age', 'Fare']})
        return dataFrame

class CleanData():
    def __getRemColsList(self, dataFrame, colsList):
        return list(filter(lambda x: x not in colsList, dataFrame.columns.values.tolist()))
    
    def __getDtypes(self, dataFrame):
        dtypeDict = dataFrame.dtypes.to_dict()
        dtypeDict.update((k, v.name) for k,v in dtypeDict.items())
        return dtypeDict
    
    def perform(self, dataFrame, training=True):
        medianList = ['Age']
        modeList = ['Embarked']
        remColsList = self.__getRemColsList(dataFrame, medianList + modeList)
        if(training):
            self.transformer = ColumnTransformer(transformers=[
                ('imp_median', SimpleImputer(strategy='median'), medianList),
                ('imp_mode', SimpleImputer(strategy='most_frequent'), modeList)
            ],remainder='passthrough')
            self.transformer.fit(dataFrame)
            self.newColsList = medianList + modeList + remColsList
        transformedNpArr = self.transformer.transform(dataFrame)
        tempDF = pd.DataFrame(transformedNpArr, columns=self.newColsList)
        dataFrame = tempDF.astype(self.__getDtypes(dataFrame))
        return dataFrame
    
class CategoryConvert():
    def __getRemColsList(self, dataFrame, colsList):
        return list(filter(lambda x: x not in colsList, dataFrame.columns.values.tolist()))
     
    def __getDtypes(self, dataFrame):
        dtypeDict = dataFrame.dtypes.to_dict()
        dtypeDict.update((k, v.name) for k,v in dtypeDict.items())
        return dtypeDict
    
    def perform(self, dataFrame, training=True):
        colsList = dataFrame.select_dtypes(include='category').columns.values.tolist()
        remColsList = self.__getRemColsList(dataFrame, colsList)
        if(training):
            self.transformer = ColumnTransformer(transformers=[
                ('onehot', OneHotEncoder(sparse=False), colsList)
            ],remainder='passthrough')
            self.transformer.fit(dataFrame)
            self.newColsList = list(self.transformer.transformers_[0][1].get_feature_names(colsList)) + remColsList
        transformedNpArr = self.transformer.transform(dataFrame)
        tempDF = pd.DataFrame(transformedNpArr, columns=self.newColsList)
        dataFrame = tempDF.astype(self.__getDtypes(dataFrame[remColsList]))
        return dataFrame

class ScaleFeature():
    def perform(self, dataFrame, training=True):
        if(training):
            self.transformer = StandardScaler()
            self.transformer.fit(dataFrame)   
        npOutput = self.transformer.transform(dataFrame)
        return npOutput

class PreProcess():
    def __init__(self):
        self.enhance = Enhance()
        self.cleanData = CleanData()
        self.categoryConvert = CategoryConvert()
        self.scaleFeature = ScaleFeature()

    def perform(self, dataFrame, training=True):
        dataFrame = self.enhance.setDtypes(dataFrame, {'category':['Pclass', 'Sex', 'Embarked'], 'string':['Name']})
        dataFrame = self.cleanData.perform(dataFrame, training)
        dataFrame = self.enhance.perform(dataFrame, training)
        dataFrame = self.categoryConvert.perform(dataFrame, training)
        npOutput = self.scaleFeature.perform(dataFrame, training)
        return npOutput

class SelectFeatures():
    def perform(self, model, attrs, labels=None, training=True):
        if(training):
            minFeatures = int(len(list(attrs[0]))*1.0)
            self.selector = RFECV(model, cv=5, scoring="accuracy", n_jobs=-1, min_features_to_select=minFeatures)
            self.selector.fit(attrs, labels)
        return self.selector.transform(attrs)
        
class ValidateModels():
    def __init__(self, split):
        self.split = split
        
    def perform(self, model, attrSet, labelSet, classDist='uniform'):
        if(classDist == 'uniform'):
            scores = cvs(model, attrSet, labelSet, scoring="accuracy", cv=self.split)
            print("Accuracy Estimate: MEAN +/- STD = " + str(np.round(scores.mean(),3)) + " +/- " + str(np.round(scores.std(),3)))
        elif(classDist == 'skewed'):
            preds = cvp(model, attrSet, labelSet, cv=self.split)
            print("Confusion Matrix: ")
            print(metrics.confusion_matrix(labelSet, preds))
            print("Classification report: ")
            print(metrics.classification_report(labelSet, preds))

class FineTune():
    def __init__(self, type='randomized'):
        self.type = type
        
    def perform(self, model, param_grid, attrSet, labelSet):
        if(self.type == 'grid'):
            search = GridSearchCV(model, param_grid, cv=5, scoring="accuracy")
        elif(self.type == 'randomized'):
            n_iter_search = 25
            search = RandomizedSearchCV(model, param_grid, n_iter_search, cv=5, scoring='accuracy')
        search.fit(attrSet, labelSet)
        print()
        print("*****==========*****")
        print("Best Model: " + str(search.best_estimator_))
        print("*****==========*****")
        print()
        return (search.best_params_, search.best_estimator_)

class Test():
    def perform(self, model, attrSet, labelSet=None):
        predictions = model.predict(attrSet)
        # print()
        # print("*****==========*****")
        # print("Accuracy Score: " + str(metrics.accuracy_score(labelSet, predictions)))
        # print("*****==========*****")
        # print("Confusion Matrix: ")
        # print(metrics.confusion_matrix(labelSet, predictions))
        # print("*****==========*****")
        # print("Classification report: ")
        # print(metrics.classification_report(labelSet, predictions))
        # print("*****==========*****")
        # print()
        return predictions
    
### ==== ACTUAL IMPLEMENTATION ==== ###

preProcess = PreProcess()
validateModels = ValidateModels(10)
selectFeatures = SelectFeatures()
fineTune = FineTune('grid')

impCols = ['Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
trainSet = pd.read_csv("Train_Subset.csv", sep=",", usecols=impCols)


labels = ['Survived']
trainLabels = trainSet[labels].astype(float).astype('category')
trainAttrs = trainSet.drop(labels, axis=1)

X_train = preProcess.perform(trainAttrs)
Y_train = trainLabels.values.ravel()

dtForFS = tree.DecisionTreeClassifier(random_state=50, splitter='random', max_depth=4, criterion='entropy')
# param_grid = [
#         {"criterion" : ["gini", "entropy"],
#          "splitter" : ["best", "random"],
#          "max_depth" : list(np.linspace(1,10, num=10, dtype=int)) + [None],
#          "min_samples_split" : list(map(lambda x: round(x,3), np.linspace(0.05,1, num=20))) + [2],
#          "max_features" : list(map(lambda x: round(x,3), np.linspace(0.05,1, num=21))) + ["auto", "sqrt", "log2", None]
#         }
#     ]
# bestParams, bestModel = fineTune.perform(dtForFS, param_grid, X_train, Y_train)
# print(bestParams)
# validateModels.perform(bestModel, X_train, Y_train)
X_train = selectFeatures.perform(dtForFS, X_train, labels=Y_train)
print(X_train[0])

knn_model = neighbors.KNeighborsClassifier(n_jobs=-1)
lr_model = linear_model.LogisticRegression(random_state=50, n_jobs=-1)
dt_model = tree.DecisionTreeClassifier(random_state=50)
rf_model = ensemble.RandomForestClassifier(random_state=50, n_jobs=-1)
bc_model = ensemble.BaggingClassifier(random_state=50, n_jobs=-1)
abc_model = ensemble.AdaBoostClassifier(random_state=50)
svc_model = svm.SVC(random_state=50, C=0.5, gamma='auto', probability=True)
et_model = ensemble.ExtraTreesClassifier(random_state=50, n_jobs=-1, n_estimators=50, criterion='entropy', max_depth=5)
gbc_model = ensemble.GradientBoostingClassifier(random_state=50, loss="exponential", n_estimators=750, n_iter_no_change=2, criterion="friedman_mse", init= tree.DecisionTreeClassifier(max_depth=1, criterion='entropy',random_state=50, splitter='best'))
vc_model = ensemble.VotingClassifier(estimators=[('svc', svc_model), ('et', et_model), ('gbc', gbc_model), ('knn', knn_model), ('lr', lr_model), ('dt', dt_model), ('rf', rf_model), ('bc', bc_model), ('abc', abc_model), ('dtForFS', dtForFS)], voting='soft', n_jobs=-1)
vc_model.fit(X_train, Y_train)

impCols = ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
ids = ['PassengerId']
testSet = pd.read_csv("Test_Subset.csv", sep=",", usecols=impCols)
testIds = testSet[ids].astype(int)
testAttrs = testSet.drop(ids, axis=1)
X_test = preProcess.perform(testAttrs, training=False)
X_test = selectFeatures.perform(dtForFS, X_test, training=False)
print(X_test[0])

test = Test()
preds = test.perform(vc_model, X_test)
testIds['Survived'] = pd.Series(preds, dtype=int)
testIds.to_csv('myPrediction.csv', index=False)