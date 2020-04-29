# California Housing -- Pricing Prediction -- Dataset Used to Present the Regression Template

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
from sklearn import tree
from sklearn import ensemble
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error as mse
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import cross_val_score as cvs
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from pandas.plotting import scatter_matrix

class BirdsEyeView(): #Used to get a high-level information about how the dataset looks.
    def __init__(self, dataFrame):
        self.dataFrame = dataFrame
    
    def getSample(self, rows=10):
        return self.dataFrame.head(rows)
    
    def getInfo(self):
        self.dataFrame.info()
    
    def getNumAttrStats(self):
        return self.dataFrame.describe()
    
    def plotAttrHist(self, bins=50):
        self.dataFrame.hist(bins=bins)
        plt.show()

class Explore(): #Used to get further insights regarding the various attributes and features of the dataset.
    def __init__(self, dataFrame):
        self.dataFrame = dataFrame
    
    # USECASE SPECIFIC IMPLEMENTATION
    def getCorrToLabels(self, labels):
        corrMatrix = self.dataFrame.corr()
        return corrMatrix[labels].sort_values(ascending=True)
    
    def plotScatterMatrix(self, attrs):
        scatter_matrix(self.dataFrame[attrs])
        plt.show()
    # Can add other exploration algorithms in order to get further insights regarding the dataset.

class Split(): #Used to split the dataset into training and testing subsets. Stratified Split also implemented.
    def __init__(self, test_size, labels, stratify=False, stratifyBy=None):
        self.test_size = test_size
        self.labels = labels
        self.stratify = stratify
        self.stratifyBy = stratifyBy
    
    # USECASE SPECIFIC IMPLEMENTATION                
    def __getCategory(self, desDF, colName, val):
        cat25 = desDF[colName].loc['25%']
        cat50 = desDF[colName].loc['50%']
        cat75 = desDF[colName].loc['75%']
        if(val <= cat25): return 'vl'
        elif(val <= cat50): return 'l'
        elif(val <= cat75): return 'h'
        else: return 'vh'

    def __getQuartile_Category(self, dataFrame):
        desDF = dataFrame.describe()
        catColNames = []
        for i in self.stratifyBy:
            catColNames.append("cat_"+i)
            dataFrame["cat_"+i] = dataFrame[i].apply(lambda x: self.__getCategory(desDF, i, x))
        return dataFrame, catColNames
    # Can add other algorithms to generate categorical values from continious values to enable stratification, based on usecase.

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
            dataFrame, catCols = self.__getQuartile_Category(dataFrame)
            return self.__get_tts_from_df(dataFrame, self.test_size, attrs, self.labels, dataFrame[catCols])
        else:
            return self.__get_tts_from_df(dataFrame, self.test_size, attrs, self.labels)

class Enhance(): #PreProcessing Step-1: The dataset is enhanced (manipulated) here so that it can provide more & useful information, based on usecase. 
    def perform(self, dataFrame):
        # USECASE SPECIFIC IMPLEMENTATION
        dataFrame['avgRooms_per_household'] = dataFrame['total_rooms']/dataFrame['households']
        dataFrame['avgBedRooms_per_room'] = dataFrame['total_bedrooms']/dataFrame['total_rooms']
        # Can tranform dataframe as needed to increase the information it provides.
        return dataFrame

class CleanData(): #PreProcessing Step-2: The dataset is cleaned here so that no missing values hinder the performance using algorithms, based on usecase.
    def __getColLocs(self, dataFrame, colsList):
        return list(map(lambda x: dataFrame.columns.get_loc(x), colsList))
    
    def perform(self, dataFrame, training=True):
        colsList = dataFrame.describe().columns.values.tolist()
        remColsList = list(set(dataFrame.columns.values.tolist())-set(colsList))
        if(training):
            # USECASE SPECIFIC IMPLEMENTATION
            self.transformer = ColumnTransformer(transformers=[
                ('imp_median', SimpleImputer(strategy='median'), self.__getColLocs(dataFrame, colsList))
            ],remainder='passthrough')
            # Can implement different type of ColumnTransformer, for various number of columns which require cleaning.
            self.transformer.fit(dataFrame)
        transformedNpArr = self.transformer.transform(dataFrame)
        dataFrame = pd.DataFrame(transformedNpArr, columns=colsList+remColsList)
        return dataFrame

class ColumnLabelBinarizer(BaseEstimator, TransformerMixin): #Used by PreProcessing Step-3.
    def __init__(self):
        self.encoder = LabelBinarizer()
    def fit(self, x, y=None):
        self.encoder.fit(x)
        return self
    def transform(self, x, y=None):
        return self.encoder.transform(x)
     
class CategoryConvert(): #PreProcessing Step-3: The categorical values in the dataset are converted to binary labels using algorithms, based on usecase.
    def __getColLocs(self, dataFrame, colsList):
        return list(map(lambda x: dataFrame.columns.get_loc(x), colsList))
    
    def perform(self, dataFrame, training=True):
        colsList = ['ocean_proximity']
        remColsList = list(set(dataFrame.columns.values.tolist())-set(colsList))
        if(training):
            # USECASE SPECIFIC IMPLEMENTATION
            self.transformer = ColumnTransformer(transformers=[
                ('label_binarizer', ColumnLabelBinarizer(), self.__getColLocs(dataFrame, colsList))
            ],remainder='passthrough')
            # Can implement different type of ColumnTransformer, for various number of columns which require conversion.
            self.transformer.fit(dataFrame)
        transformedNpArr = self.transformer.transform(dataFrame)
        dataFrame = pd.DataFrame(transformedNpArr)
        return dataFrame

class ScaleFeature(): #PreProcessing Step-4: All the values in the dataset are scaled for better performance using algorithms, based on usecase.
    def perform(self, dataFrame, training=True):
        if(training):
            self.transformer = StandardScaler()
            self.transformer.fit(dataFrame)   
        npOutput = self.transformer.transform(dataFrame)
        return npOutput

class PreProcess(): #PreProcessing PipeLine : Combines all the Pre-Processing Steps.
    def __init__(self):
        self.enhance = Enhance()
        self.cleanData = CleanData()
        self.categoryConvert = CategoryConvert()
        self.scaleFeature = ScaleFeature()

    def perform(self, dataFrame, training=True):
        # USECASE SPECIFIC IMPLEMENTATION
        dataFrame = self.enhance.perform(dataFrame)
        dataFrame = self.cleanData.perform(dataFrame, training)
        dataFrame = self.categoryConvert.perform(dataFrame, training)
        npOutput = self.scaleFeature.perform(dataFrame, training)
        # Can chuck the pipeline and perform each step individually for more control over manipulations, also can combine steps into one (forEg: Cleaning and Conversion) if it simplfies/optimizes the task.
        return npOutput

class ValidateModels(): #Validate Different ML Models to shortlist the best one, the scoring metric can be altered, bassed on usecase.
    def perform(self, model, attrSet, labelSet):
        scores = cvs(model, attrSet, labelSet, scoring="neg_mean_squared_error", cv=10)
        scores = np.sqrt(-scores)
        print("Cost Function Estimate: MEAN +/- STD = " + str(np.round(scores.mean(),3)) + " +/- " + str(np.round(scores.std(),3)))

class FineTune(): #Find the best hyperparamters for the shortliested ML Model, the scoring metric can be altered, based on usecase.
    def __init__(self, type='randomized'):
        self.type = type
        
    def perform(self, model, param_grid, attrSet, labelSet):
        if(self.type == 'grid'):
            search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
        elif(self.type == 'randomized'):
            n_iter_search = 20
            search = RandomizedSearchCV(model, param_grid, n_iter_search, cv=5, scoring='neg_mean_squared_error')
        search.fit(attrSet, labelSet)
        print()
        print("*****==========*****")
        print("Best Model: " + str(search.best_estimator_))
        print("*****==========*****")
        print()
        return (search.best_params_, search.best_estimator_, search.best_estimator_.feature_importances_)

class ExtractImpFeats(): #Remove Noisy Features from the attributes in order the further enhance the performance of the model.
    def __init__(self, featureImps, numOfFeats):
        self.featureImps = featureImps
        self.numOfFeats = numOfFeats
        
    def perform(self,  bestModel=None, trainAttrs=None, trainLabels=None, testAttrs=None, training=True):
        indices = np.sort(np.argpartition(np.array(self.featureImps), -self.numOfFeats)[-self.numOfFeats:])
        if(training):
            trainAttrs = trainAttrs[: , indices]
            bestModel.fit(trainAttrs, trainLabels)
            return bestModel
        else:
            testAttrs = testAttrs[: , indices]
            return testAttrs

class Test(): #Test the final model on the Testing Subset and evaluate performance.
    def perform(self, model, attrSet, labelSet):
        predictions = model.predict(attrSet)
        rmse = np.sqrt(mse(labelSet, predictions))
        print()
        print("*****==========*****")
        print("RMSE for the Best Model on Test Set: " + str(np.round(rmse,3)))
        print("*****==========*****")
        print()
        return predictions

### ==== ACTUAL IMPLEMENTATION ==== ###            
caliHousing = pd.read_csv("californiaHousing-price.csv", sep=",")

# Quick Look at the Structure of the Dataset
birdsEyeView = BirdsEyeView(caliHousing)
print(birdsEyeView.getSample())
birdsEyeView.getInfo()
print(birdsEyeView.getNumAttrStats())
birdsEyeView.plotAttrHist()

# Splitting Dataset into Training and Testing Subsets
labels = ['median_house_value']
split = Split(0.2, labels, True, ['median_income'] + labels)
trainDF, testDF, Y_train, Y_test = split.perform(caliHousing)

#Data Exploration
explore = Explore(trainDF)
for i in labels:
  print(explore.getCorrToLabels(i))

#PreProcessing the Subsets
preProcess = PreProcess()
X_train = preProcess.perform(trainDF)

#Initializing Models
lin_reg_model = linear_model.LinearRegression()
tree_reg_model = tree.DecisionTreeRegressor()
forest_reg_model = ensemble.RandomForestRegressor()

#Cross-Validating the Models.
validateModels = ValidateModels()
validateModels.perform(lin_reg_model, X_train, Y_train) 
validateModels.perform(tree_reg_model, X_train, Y_train)
validateModels.perform(forest_reg_model, X_train, Y_train.ravel())
# forest_reg_model Results are Best.

#Fine Tuninng the Best Model.
fineTune = FineTune('grid')
param_grid = [
        {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
        {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
    ]
bestParams, bestModel, featureImps = fineTune.perform(forest_reg_model, param_grid, X_train, Y_train)
# bestModel is the Final Model to be used for Evaluation on Test Set.

#Extracting Important Features for bestModel
extractImpFeats = ExtractImpFeats(featureImps, 8)
bestModel = extractImpFeats.perform(bestModel, X_train, Y_train)

#Keeping only the Important Features for bestModel from the test attributes.
X_test = preProcess.perform(testDF, False)
X_test = extractImpFeats.perform(testAttrs=X_test, training=False)
#Predicting Labels for Testing Subset
test = Test()
predictions = test.perform(bestModel, X_test, Y_test)