import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as tts

class Utils:
    def getNAStats(self, dataFrame):
        return dataFrame.isna().sum()

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
    def perform(self, dataFrame):
        categoryCols = ['Pclass', 'Sex', 'Embarked']
        for i in categoryCols:
            dataFrame[i] = dataFrame[i].astype('category')
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