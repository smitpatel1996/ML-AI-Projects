import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import mean_squared_error as mse
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
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

    def __getQuartile_Category(self, dataFrame, colsList):
        desDF = dataFrame.describe()
        catColNames = []
        for i in colsList:
            catColNames.append("cat_"+i)
            dataFrame["cat_"+i] = dataFrame[i].apply(lambda x: self.__getCategory(desDF, i, x))
        return dataFrame, catColNames

    def __get_tts_from_df(self, dataFrame, test_size, attrs, labels, stratifyBy=None):
        if(self.stratify):
            train, test = tts(dataFrame, test_size=test_size, stratify=stratifyBy, random_state=50)
        else:
            train, test = tts(dataFrame, test_size=test_size, random_state=50)
        X_train = (train[attrs]).to_numpy().reshape(-1,1)
        Y_train = (train[labels]).to_numpy().reshape(-1,1)
        X_test = (test[attrs]).to_numpy().reshape(-1,1)
        Y_test = (test[labels]).to_numpy().reshape(-1,1)
        return (train[attrs+labels], test[attrs+labels], X_train, X_test, Y_train, Y_test)
    
    def perform(self, dataFrame):
        labels = self.labels
        attrs = dataFrame.drop(labels, axis=1).columns.values.tolist()
        if(self.stratify):
            dataFrame, catCols = self.__getQuartile_Category(dataFrame, self.stratifyBy)
            return self.__get_tts_from_df(dataFrame, self.test_size, attrs, labels, dataFrame[catCols])
        else:
            return self.__get_tts_from_df(dataFrame, self.test_size, attrs, labels)

class Enhance():
    def __init__(self, roomsPerHouse=True, bedroomsPerRoom=True):
        self.roomsPerHouse = roomsPerHouse
        self.bedroomsPerRoom = bedroomsPerRoom
        
    def perform(self, dataFrame):
        if(self.roomsPerHouse):
            dataFrame['avgRooms_per_household'] = dataFrame['total_rooms']/dataFrame['households']
        if(self.bedroomsPerRoom):
            dataFrame['avgBedRooms_per_room'] = dataFrame['total_bedrooms']/dataFrame['total_rooms']
        return dataFrame

class CleanData():
    def __init__(self, strategy='mean', missing_values=np.nan, training=True):
        self.imp = SimpleImputer(missing_values=missing_values, strategy=strategy)
        self.strategy = strategy
        self.training = training
    
    def perform(self, dataFrame):
        if(self.strategy == 'median' or self.strategy == 'mean'):
            numCols = dataFrame.describe().columns.values.tolist()
            remCols = list(set(dataFrame.columns.values.tolist())-set(numCols))
            remDf = dataFrame[remCols]
            remDf.reset_index(drop=True, inplace=True)
            dataFrame = dataFrame[numCols]
            if(self.training):            
                self.imp.fit(dataFrame)
            filledNpArr = self.imp.transform(dataFrame)
            dataFrame = pd.DataFrame(filledNpArr, columns=dataFrame.columns)
            dataFrame = pd.concat([dataFrame, remDf], axis=1)
        return dataFrame
        
class CategoryConvert():
    def __init__(self, sparse=False, training=True):
        self.encoder = LabelBinarizer(sparse_output=sparse)
        self.training = training
    
    def perform(self, dataFrame, catAttrs):
        remCols = dataFrame.drop(catAttrs, axis=1).columns.values.tolist()
        remDf = dataFrame[remCols]
        remDf.reset_index(drop=True, inplace=True)
        dataFrame = dataFrame[catAttrs]
        if(self.training):
            self.encoder.fit(dataFrame)
        dataFrame = pd.DataFrame(self.encoder.transform(dataFrame), columns= self.encoder.classes_)
        dataFrame = pd.concat([remDf, dataFrame], axis=1)
        return dataFrame    

class ScaleFeature():
    def __init__(self, training=True):
        self.scaler = StandardScaler()
        self.training = training
    
    def perform(self, dataFrame):
        if(self.training):
            self.scaler.fit(dataFrame)   
        dataFrame = pd.DataFrame(self.scaler.transform(dataFrame) , columns= dataFrame.columns.values.tolist())      
        return dataFrame

class PrePorcess():
    def __init__(self, training=True):
        self.training = training
        self.enhance = Enhance()
        self.cleanData = CleanData(strategy='median', training=training)
        self.categoryConvert = CategoryConvert(training=training)
        self.scaleFeature = ScaleFeature(training=training)

    def perform(self, dataFrame):
        dataFrame = self.enhance.perform(dataFrame)
        dataFrame = self.cleanData.perform(dataFrame)
        dataFrame = self.categoryConvert.perform(dataFrame, 'ocean_proximity')
        dataFrame = self.scaleFeature.perform(dataFrame)
        return dataFrame
        
caliHousing = pd.read_csv("californiaHousing-price.csv", sep=",")

# Quick Look at the Structure of the Dataset
birdsEyeView = BirdsEyeView(caliHousing)
#print(birdsEyeView.getSample())
#birdsEyeView.getInfo()
#print(birdsEyeView.getNumAttrStats())
#birdsEyeView.plotAttrHist()

# Splitting Dataset into Training and Testing Subsets
labels = ['median_house_value']
split = Split(0.2, labels, True, ['median_income'] + labels)
trainDF, testDF, X_train, X_test, Y_train, Y_test = split.perform(caliHousing)

#Data Exploration
explore = Explore(trainDF)
# for i in labels:
#     print(explore.getCorrToLabels(i))

#PreProcessing Training Subset
preProcess = PrePorcess()
trainDF = preProcess.perform(trainDF)
print(trainDF.head())