import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

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

class MCDropout(keras.layers.Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)

class MCAlphaDropout(keras.layers.AlphaDropout):
    def call(self, inputs):
        return super().call(inputs, training=True)

class NeuralNet():    
    def __inputLayer(self, name, X_train):
        return keras.layers.Input(shape=X_train.shape[1:], name=name)
    
    def __hiddenLayer(self, neurons, actFunc, kernelInit=None, kernelReg=None, kernelConst=None):
        return keras.layers.Dense(neurons, activation=actFunc, kernel_initializer=kernelInit, kernel_regularizer=kernelReg, kernel_constraint=kernelConst)
    
    def __outputLayer(self, name, neurons, actFunc):
        return keras.layers.Dense(neurons, activation=actFunc, name=name)
    
    def build(self, X_train, model='Sequential'):
        if(model == 'Sequential'):
            self.model = keras.models.Sequential()
            self.model.add(self.__inputLayer('input', X_train))
            self.model.add(self.__hiddenLayer(392, 'selu', 'lecun_normal', keras.regularizers.l1(0.0001)))
            self.model.add(self.__hiddenLayer(392, 'selu', 'lecun_normal', kernelConst=keras.constraints.MaxNorm(1000.)))
            self.model.add(self.__hiddenLayer(196, 'selu', 'lecun_normal', keras.regularizers.l1(0.0001)))
            self.model.add(self.__hiddenLayer(196, 'selu', 'lecun_normal', kernelConst=keras.constraints.MaxNorm(1000.)))
            self.model.add(self.__hiddenLayer(98, 'selu', 'lecun_normal', keras.regularizers.l1(0.0001)))
            self.model.add(self.__hiddenLayer(98, 'selu', 'lecun_normal', kernelConst=keras.constraints.MaxNorm(1000.)))
            self.model.add(self.__hiddenLayer(49, 'selu', 'lecun_normal', keras.regularizers.l1(0.0001)))
            self.model.add(self.__hiddenLayer(49, 'selu', 'lecun_normal', kernelConst=keras.constraints.MaxNorm(1000.)))
            self.model.add(self.__hiddenLayer(25, 'selu', 'lecun_normal'))
            self.model.add(MCAlphaDropout(rate=0.25))
            self.model.add(self.__hiddenLayer(25, 'selu', 'lecun_normal'))
            self.model.add(MCAlphaDropout(rate=0.25))
            self.model.add(self.__outputLayer('output', 10, 'softmax'))
    
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
                
    def compile(self, opt):
        #learning_rate = keras.optimizers.schedules.ExponentialDecay(0.001, 5, 0.999)
        if(opt == 'Nesterov'):
            optimizer = keras.optimizers.SGD(learning_rate=0.025, momentum=0.9, nesterov=True)
        if(opt == 'RMSprop'):
            optimizer = keras.optimizers.RMSprop(learning_rate=0.0025, rho=0.9)
        self.model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    
    def fit(self, trainSet, valSet, epochs):
        X_train, Y_train = trainSet
        X_valid, Y_valid = valSet
        save_best = keras.callbacks.ModelCheckpoint("MNIST-NN.h5", save_best_only=True)
        early_stop = keras.callbacks.EarlyStopping(patience=int(0.2*epochs), restore_best_weights=True)
        lr_scheduler = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=int(0.075*epochs))
        self.history = self.model.fit(X_train, Y_train, epochs=epochs, validation_data=(X_valid, Y_valid), batch_size=50, callbacks=[save_best, early_stop, lr_scheduler])
    
    def plotLearningCurve(self):
        pd.DataFrame(self.history.history).plot()
        plt.grid(True)
        plt.show()
    
    def assemble(self, trainSet, valSet, epochs):
        X_train, Y_train = trainSet
        X_valid, Y_valid = valSet
        self.build(X_train)
        self.get_Info('summary')
        self.compile('Nesterov')
        self.fit(trainSet, valSet, epochs)
    
    def evaluate(self, testSet):
        (X_test, Y_test) = testSet
        self.model.evaluate(X_test, Y_test)
    
    def predict(self, newFeVec, mcDrop=True):
        if(mcDrop):
            preds = np.stack([self.model.predict(newFeVec) for i in range(10)])
            preds = np.round(((preds).mean(axis=0)),2)
            print("Probabilities:", preds)
        else:
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

print(X_train.shape)
print(Y_train.shape)
neuralNet = NeuralNet()
neuralNet.assemble((X_train, Y_train), (X_valid, Y_valid), 100)
neuralNet.plotLearningCurve()

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
print("Final Classes: ", preds)
print(preds.shape)
testIds['Label'] = pd.Series(preds, dtype=int)
testIds.to_csv('myPrediction.csv', index=False)