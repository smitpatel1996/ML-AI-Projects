import random
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn import tree
from sklearn import metrics
from sklearn import ensemble
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
    def perform(self, model, attrs, labels, training=True):
        if(training):
            minFeatures = int(len(list(attrs[0]))*0.25)
            self.selector = RFECV(model, cv=5, scoring="accuracy", n_jobs=-1, min_features_to_select=minFeatures)
            self.selector.fit(attrs, labels)
        return (self.selector.transform(attrs), labels)
        
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
            n_iter_search = 100
            search = RandomizedSearchCV(model, param_grid, n_iter_search, cv=5, scoring='accuracy')
        search.fit(attrSet, labelSet)
        print()
        print("*****==========*****")
        print("Best Model: " + str(search.best_estimator_))
        print("*****==========*****")
        print()
        return (search.best_params_, search.best_estimator_)


### ==== ACTUAL IMPLEMENTATION ==== ###

impCols = ['Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
trainSet = pd.read_csv("Train_Subset.csv", sep=",", usecols=impCols)

labels = ['Survived']
trainLabels = trainSet[labels].astype(float).astype('category')
trainAttrs = trainSet.drop(labels, axis=1)

preProcess = PreProcess()
X_train = preProcess.perform(trainAttrs)
Y_train = trainLabels.values.ravel()

dtForFS = tree.DecisionTreeClassifier(random_state=50)
selectFeatures = SelectFeatures()
X_train, Y_train = selectFeatures.perform(dtForFS, X_train, Y_train)
print(X_train[0])

svm_model = svm.SVC(random_state=50)
rf_model = ensemble.RandomForestClassifier(random_state=50, n_jobs=-1)
gdb_model = ensemble.GradientBoostingClassifier(random_state=50)

validateModels = ValidateModels(10)
validateModels.perform(svm_model, X_train, Y_train)
validateModels.perform(rf_model, X_train, Y_train)
validateModels.perform(gdb_model, X_train, Y_train)