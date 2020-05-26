import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

class NeuralNet():
    def __inputLayer(self, name, X_train):
        return keras.layers.Flatten(input_shape=X_train.shape[1:], batch_size=20, name=name)
    
    def __hiddenLayer(self, neurons, actFunc):
        return keras.layers.Dense(neurons, activation=actFunc)
    
    def __outputLayer(self, name, neurons, actFunc=None):
        return keras.layers.Dense(neurons, activation=actFunc, name=name)
    
    def build(self, X_train):
        self.model = keras.models.Sequential()
        self.model.add(self.__inputLayer('input', X_train))
        self.model.add(self.__hiddenLayer(400, 'relu'))
        self.model.add(self.__hiddenLayer(200, 'relu'))
        self.model.add(self.__hiddenLayer(50, 'relu'))
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
                
    def compile(self):
        optimizer = keras.optimizers.SGD(lr=0.01)
        self.model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    
    def fit(self, trainSet, valSet, epochs):
        X_train, Y_train = trainSet
        X_valid, Y_valid = valSet
        self.history = self.model.fit(X_train, Y_train, epochs=epochs, validation_data=(X_valid, Y_valid))
    
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
        self.history = self.model.fit(X_train, Y_train, epochs=epochs)
    
    def evaluate(self, testSet):
        (X_test, Y_test) = testSet
        self.model.evaluate(X_test, Y_test)
    
    def predict(self, newFeVec, probability=False):
        return self.model.predict_classes(newFeVec, batch_size=20)
                    
f_mnist = keras.datasets.fashion_mnist
(X_train_full, Y_train_full), (X_test, Y_test) = f_mnist.load_data()
f_classNames = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
Y_valid, Y_train = Y_train_full[:5000], Y_train_full[5000:]
X_train_full = X_train_full / 255.0

neuralNet = NeuralNet()
neuralNet.assemble((X_train, Y_train), (X_valid, Y_valid), 30)
neuralNet.plotLearningCurve()
neuralNet.compose((X_train_full, Y_train_full), 30)
neuralNet.evaluate((X_test, Y_test))
preds = neuralNet.predict(X_test[:3])
print(preds)
print(Y_test[:3])