import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as tts

class ScaleFeature():
    def perform(self, dataFrame, training=True):
        if(training):
            self.transformer = StandardScaler()
            self.transformer.fit(dataFrame)   
        npOutput = self.transformer.transform(dataFrame)
        return npOutput

class PreProcess():
    def __init__(self):
        self.scaleFeature = ScaleFeature()

    def perform(self, dataFrame, training=True):
        npOutput = self.scaleFeature.perform(dataFrame, training)
        return npOutput

class ValidationSplit():
    def __init__(self, val_size):
        self.val_size = val_size
    
    def perform(self, attrs, labels):
        index = np.random.choice(attrs.shape[0], int(len(attrs)*(1-self.val_size)), replace=False)
        return (np.take(attrs, index, axis=0), np.delete(attrs, index, axis=0), np.take(labels, index, axis=0), np.delete(labels, index, axis=0))
 
class NeuralNet():
    def __inputLayer(self, name, X_train):
        return keras.layers.Input(shape=X_train.shape[1:], name=name)
    
    def __hiddenLayer(self, neurons, actFunc):
        return keras.layers.Dense(neurons, activation=actFunc)
    
    def __outputLayer(self, name, neurons, actFunc=None):
        return keras.layers.Dense(neurons, activation=actFunc, name=name)
    
    def __concatLayer(self, layerList):
        return keras.layers.Concatenate()(layerList)
    
    def build(self, X_train):
        inputLayer = self.__inputLayer('input', X_train)
        hiddenLayer1 = (self.__hiddenLayer(392, 'relu'))(inputLayer)
        hiddenLayer2 = (self.__hiddenLayer(392, 'relu'))(hiddenLayer1)
        hiddenLayer3 = (self.__hiddenLayer(196, 'relu'))(hiddenLayer2)
        hiddenLayer4 = (self.__hiddenLayer(196, 'relu'))(hiddenLayer3)
        hiddenLayer5 = (self.__hiddenLayer(49, 'tanh'))(inputLayer)
        hiddenLayer6 = (self.__hiddenLayer(49, 'tanh'))(hiddenLayer5)
        concatLayer = self.__concatLayer([hiddenLayer4, hiddenLayer6])
        outputlayer = (self.__outputLayer('output', 10, 'softmax'))(concatLayer)
        self.model = keras.Model(inputs=[inputLayer], outputs=[outputlayer])
    
    def get_Info(self, info):
        if(info == "summary"):
            self.model.summary()
        if(info == "layer"):
            for i in self.model.layers[1:]:
                weights, biases = i.get_weights()
                print("====================")
                print("Layer Name:", i.name)
                print("Layer Connection Weights Shape:", weights.shape)
                print("Layer Connection Weights:", weights)
                print("Layer Biases Shape:", biases.shape)
                print("Layer Biases:", biases)
                print("====================")
                
    def compile(self):
        optimizer = keras.optimizers.SGD(lr=0.05)
        self.model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    
    def fit(self, trainSet, valSet, epochs):
        X_train, Y_train = trainSet
        X_valid, Y_valid = valSet
        self.history = self.model.fit(X_train, Y_train, epochs=epochs, validation_data=(X_valid, Y_valid), batch_size=32)
    
    def plotLearningCurve(self):
        pd.DataFrame(self.history.history).plot()
        plt.grid(True)
        plt.show()
    
    def assemble(self, trainSet, valSet, epochs):
        X_train, Y_train = trainSet
        X_valid, Y_valid = valSet
        self.build(X_train)
        self.get_Info('summary')
        self.compile()
        self.fit(trainSet, valSet, epochs)
    
    def compose(self, trainSet, epochs):
        X_train, Y_train = trainSet
        self.build(X_train)
        self.compile()
        self.history = self.model.fit(X_train, Y_train, epochs=epochs, batch_size=32)
    
    def evaluate(self, testSet):
        (X_test, Y_test) = testSet
        self.model.evaluate(X_test, Y_test)
    
    def predict(self, newFeVec, probability=False):
        preds = self.model.predict(newFeVec)
        return preds.argmax(axis=-1)

### ==== ACTUAL IMPLEMENTATION ==== ###
                  
trainSet = pd.read_csv("train.csv", sep=",")
labels = ['label']
trainLabels = trainSet[labels].astype(int).astype('category')
trainAttrs = trainSet.drop(labels, axis=1)

preProcess = PreProcess()
X_train_full = preProcess.perform(trainAttrs)
Y_train_full = trainLabels.values.ravel()

validationSplit = ValidationSplit(0.2)
X_train, X_valid, Y_train, Y_valid = validationSplit.perform(X_train_full, Y_train_full)

neuralNet = NeuralNet()
neuralNet.assemble((X_train, Y_train), (X_valid, Y_valid), 25)
neuralNet.plotLearningCurve()
print(X_train_full.shape)
print(Y_train_full.shape)
neuralNet.compose((X_train_full, Y_train_full), 25)

testSet = pd.read_csv("test.csv", sep=",")
testSet = testSet.reset_index()
testSet = testSet.rename({'index': 'ImageId'}, axis=1)
ids = ['ImageId']
testIds = testSet[ids].astype(int)
testIds.ImageId = testIds.ImageId + 1
testAttrs = testSet.drop(ids, axis=1)
X_test = preProcess.perform(testAttrs, training=False)
print(X_test.shape)
preds = neuralNet.predict(X_test)
print(preds.shape)
testIds['Label'] = pd.Series(preds, dtype=int)
testIds.to_csv('myPrediction.csv', index=False)