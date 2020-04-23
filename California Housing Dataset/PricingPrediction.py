import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error as mse
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as tts
from pandas.plotting import scatter_matrix

#Utility Classes
class BirdsEyeView():
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

class Explore():
    def __init__(self, dataFrame):
        self.dataFrame = dataFrame
    
    def getCorrToLabels(self, labels):
        corrMatrix = self.dataFrame.corr()
        return corrMatrix[labels].sort_values(ascending=True)
    
    def plotScatterMatrix(self, attrs):
        scatter_matrix(self.dataFrame[attrs])
        plt.show()

#Transformation Classes    
class Split():
    def __init__(self, test_size, labels, stratify=False, stratifyBy=None):
        self.test_size = test_size
        self.labels = labels
        self.stratify = stratify
        self.stratifyBy = stratifyBy
                    
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

    def __get_tts_from_df(self, dataFrame, test_size, attrs, labels, stratifyBy=None):
        if(self.stratify):
            train, test = tts(dataFrame, test_size=test_size, stratify=stratifyBy, random_state=50)
        else:
            train, test = tts(dataFrame, test_size=test_size, random_state=50)
        Y_train = (train[labels]).to_numpy().reshape(-1,1)
        Y_test = (test[labels]).to_numpy().reshape(-1,1)
        return (train[attrs], test[attrs], Y_train, Y_test)
    
    def perform(self, dataFrame):
        attrs = dataFrame.drop(self.labels, axis=1).columns.values.tolist()
        if(self.stratify):
            dataFrame, catCols = self.__getQuartile_Category(dataFrame)
            return self.__get_tts_from_df(dataFrame, self.test_size, attrs, self.labels, dataFrame[catCols])
        else:
            return self.__get_tts_from_df(dataFrame, self.test_size, attrs, self.labels)

class Enhance():
    def perform(self, dataFrame):
        dataFrame['avgRooms_per_household'] = dataFrame['total_rooms']/dataFrame['households']
        dataFrame['avgBedRooms_per_room'] = dataFrame['total_bedrooms']/dataFrame['total_rooms']
        return dataFrame

class CleanData():
    def __init__(self, training=True):
        self.training = training
    
    def __getColLocs(self, dataFrame, colsList):
        return list(map(lambda x: dataFrame.columns.get_loc(x), colsList))
    
    def perform(self, dataFrame):
        colsList = dataFrame.describe().columns.values.tolist()
        remColsList = list(set(dataFrame.columns.values.tolist())-set(colsList))
        if(self.training):
            self.transformer = ColumnTransformer(transformers=[
                ('imp_median', SimpleImputer(strategy='median'), self.__getColLocs(dataFrame, colsList))
            ],remainder='passthrough')
            self.transformer.fit(dataFrame)
        transformedNpArr = self.transformer.transform(dataFrame)
        dataFrame = pd.DataFrame(transformedNpArr, columns=colsList+remColsList)
        return dataFrame 

class ColumnLabelBinarizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder = LabelBinarizer()
    def fit(self, x, y=None):
        self.encoder.fit(x)
        return self
    def transform(self, x, y=None):
        return self.encoder.transform(x)
     
class CategoryConvert():
    def __init__(self, training=True):
        self.training = training
    
    def __getColLocs(self, dataFrame, colsList):
        return list(map(lambda x: dataFrame.columns.get_loc(x), colsList))
    
    def perform(self, dataFrame):
        colsList = ['ocean_proximity']
        remColsList = list(set(dataFrame.columns.values.tolist())-set(colsList))
        if(self.training):
            self.transformer = ColumnTransformer(transformers=[
                ('label_binarizer', ColumnLabelBinarizer(), self.__getColLocs(dataFrame, colsList)),
            ],remainder='passthrough')
            self.transformer.fit(dataFrame)
        transformedNpArr = self.transformer.transform(dataFrame)
        dataFrame = pd.DataFrame(transformedNpArr)
        return dataFrame    

class ScaleFeature():
    def __init__(self, training=True):
        self.training = training
           
    def perform(self, dataFrame):
        if(self.training):
            self.transformer = StandardScaler()
            self.transformer.fit(dataFrame)   
        dataFrame = self.transformer.transform(dataFrame)
        return dataFrame

class PrePorcess():
    def __init__(self, training=True):
        self.enhance = Enhance()
        self.cleanData = CleanData(training=training)
        self.categoryConvert = CategoryConvert(training=training)
        self.scaleFeature = ScaleFeature(training=training)

    def perform(self, dataFrame):
        dataFrame = self.enhance.perform(dataFrame)
        dataFrame = self.cleanData.perform(dataFrame)
        dataFrame = self.categoryConvert.perform(dataFrame)
        dataFrame = self.scaleFeature.perform(dataFrame)
        return dataFrame
        
caliHousing = pd.read_csv("californiaHousing-price.csv", sep=",")

# Quick Look at the Structure of the Dataset
#birdsEyeView = BirdsEyeView(caliHousing)
#print(birdsEyeView.getSample())
#birdsEyeView.getInfo()
#print(birdsEyeView.getNumAttrStats())
#birdsEyeView.plotAttrHist()

# Splitting Dataset into Training and Testing Subsets
labels = ['median_house_value']
split = Split(0.2, labels, True, ['median_income'] + labels)
trainDF, testDF, Y_train, Y_test = split.perform(caliHousing)

#Data Exploration
#explore = Explore(trainDF)
#for i in labels:
#   print(explore.getCorrToLabels(i))

#PreProcessing Training Subset
preProcess = PrePorcess()
X_train = preProcess.perform(trainDF)


#X_test = preProcess.perform(testDF)