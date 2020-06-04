import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from sklearn.preprocessing import StandardScaler

class Enhance():
    def __imgEnhance(self, npImg):
        return np.array(npImg).flatten()
        
    def perform(self, X_train):
        X_train = np.array(list(map(lambda x: self.__imgEnhance(x), X_train)))
        return X_train

class ScaleFeature():
    def perform(self, X_train, training=True):
        if(training):
            self.transformer = StandardScaler()
            self.transformer.fit(X_train)   
        npOutput = self.transformer.transform(X_train)
        return npOutput

class PreProcess():
    def __init__(self):
        self.enhance = Enhance()
        self.scaleFeature = ScaleFeature()

    def perform(self, X_train, training=True):
        X_train = self.enhance.perform(X_train)
        npOutput = self.scaleFeature.perform(X_train, training)
        return npOutput

class ValidationSplit():
    def __init__(self, val_size):
        self.val_size = val_size
    
    def perform(self, attrs, labels):
        index = np.random.choice(attrs.shape[0], int(len(attrs)*(1-self.val_size)), replace=False)
        return (np.take(attrs, index, axis=0), np.delete(attrs, index, axis=0), np.take(labels, index, axis=0), np.delete(labels, index, axis=0))

class NeuralNet():
    
    class __LRFinder(keras.callbacks.Callback):
        def __init__(self, min_lr, max_lr):
            self.min_lr = min_lr
            self.max_lr = max_lr
            self.mom = 0.9
            self.batches_lr_update = 5
            self.stop_multiplier = 5
        
        def on_train_begin(self, logs={}):
            p = self.params
            try:
                n_iterations = p['epochs']*p['samples']//p['batch_size']
            except:
                n_iterations = p['steps']*p['epochs']  
            self.learning_rates = np.geomspace(self.min_lr, self.max_lr, num=n_iterations//self.batches_lr_update+1)
            self.losses=[]
            self.iteration=0
            self.best_loss=0
            self.model.save_weights('tmp.hdf5')

        def on_batch_end(self, batch, logs={}):
            loss = logs.get('loss')
            if self.iteration!=0:
                loss = self.losses[-1]*self.mom+loss*(1-self.mom)
            if self.iteration==0 or loss < self.best_loss: 
                self.best_loss = loss    
            if self.iteration%self.batches_lr_update==0:            
                self.model.load_weights('tmp.hdf5')          
                lr = self.learning_rates[self.iteration//self.batches_lr_update]            
                keras.backend.set_value(self.model.optimizer.lr, lr)
                self.losses.append(loss)            
            if loss > self.best_loss*self.stop_multiplier:
                self.model.stop_training = True                
            self.iteration += 1
        
        def on_train_end(self, logs=None):
            self.model.load_weights('tmp.hdf5')        
            plt.figure(figsize=(12, 6))
            plt.plot(self.learning_rates[:len(self.losses)], self.losses)
            plt.xlabel("Learning Rate")
            plt.ylabel("Loss")
            # plt.xscale('log')
            plt.grid()
            plt.show()
            os.remove('tmp.hdf5')
    
    class __OneCycleLR(keras.callbacks.Callback):
        def __init__(self, optLR, updateMom=True, verbose=True):
            self.initial_lr = optLR
            self.end_percentage = 0.1
            if(updateMom):
                self.max_momentum = 0.95
                self.min_momentum = 0.85
                self._update_momentum = True
            else: self._update_momentum = False
            self.verbose = verbose
            self.clr_iterations = 0.
            self.history = {}

        def _reset(self):
            self.clr_iterations = 0.
            self.history = {}

        def compute_lr(self):
            if self.clr_iterations > 2 * self.mid_cycle_id:
                current_percentage = (self.clr_iterations - 2 * self.mid_cycle_id)
                current_percentage /= float((self.num_iterations - 2 * self.mid_cycle_id))
                new_lr = self.initial_lr * (1. + (current_percentage * (1. - 100.) / 100.)) * self.end_percentage
            elif self.clr_iterations > self.mid_cycle_id:
                current_percentage = 1. - (self.clr_iterations - self.mid_cycle_id) / self.mid_cycle_id
                new_lr = self.initial_lr * (1. + current_percentage * (self.end_percentage * 100 - 1.)) * self.end_percentage
            else:
                current_percentage = self.clr_iterations / self.mid_cycle_id
                new_lr = self.initial_lr * (1. + current_percentage * (self.end_percentage * 100 - 1.)) * self.end_percentage
            if self.clr_iterations == self.num_iterations:
                self.clr_iterations = 0
            return new_lr

        def compute_momentum(self):
            if self.clr_iterations > 2 * self.mid_cycle_id:
                new_momentum = self.max_momentum
            elif self.clr_iterations > self.mid_cycle_id:
                current_percentage = 1. - ((self.clr_iterations - self.mid_cycle_id) / float(self.mid_cycle_id))
                new_momentum = self.max_momentum - current_percentage * (self.max_momentum - self.min_momentum)
            else:
                current_percentage = self.clr_iterations / float(self.mid_cycle_id)
                new_momentum = self.max_momentum - current_percentage * (self.max_momentum - self.min_momentum)
            return new_momentum

        def on_train_begin(self, logs={}):
            logs = logs or {}
            self.epochs = self.params['epochs']
            self.batch_size = self.params['batch_size']
            self.samples = self.params['samples']
            self.steps = self.params['steps']
            if self.steps is not None: self.num_iterations = self.epochs * self.steps
            else:
                if (self.samples % self.batch_size) == 0: remainder = 0
                else: remainder = 1
                self.num_iterations = (self.epochs + remainder) * self.samples // self.batch_size
            self.mid_cycle_id = int(self.num_iterations * ((1. - self.end_percentage)) / float(2))
            self._reset()
            keras.backend.set_value(self.model.optimizer.lr, self.compute_lr())
            if self._update_momentum:
                new_momentum = self.compute_momentum()
                keras.backend.set_value(self.model.optimizer.momentum, new_momentum)

        def on_batch_end(self, epoch, logs=None):
            logs = logs or {}
            self.clr_iterations += 1
            new_lr = self.compute_lr()
            self.history.setdefault('lr', []).append(keras.backend.get_value(self.model.optimizer.lr))
            keras.backend.set_value(self.model.optimizer.lr, new_lr)
            if self._update_momentum:
                new_momentum = self.compute_momentum()
                self.history.setdefault('momentum', []).append(keras.backend.get_value(self.model.optimizer.momentum))
                keras.backend.set_value(self.model.optimizer.momentum, new_momentum)
            for k, v in logs.items():
                self.history.setdefault(k, []).append(v)

        def on_epoch_end(self, epoch, logs=None):
            if self.verbose:
                if self._update_momentum: print(" - lr: %0.5f - momentum: %0.2f " % (self.history['lr'][-1], self.history['momentum'][-1]))
                else: print(" - lr: %0.5f " % (self.history['lr'][-1]))
    
    class __MCDropout(keras.layers.Dropout):
        def call(self, inputs):
            return super().call(inputs, training=True)

    class __MCAlphaDropout(keras.layers.AlphaDropout):
        def call(self, inputs):
            return super().call(inputs, training=True)
    
    def __inputLayer(self, name, X_train):
        return keras.layers.Input(shape=X_train.shape[1:], name=name)
    def __hiddenLayer(self, neurons, actFunc, kernelInit=None, kernelReg=None, kernelConst=None):
        return keras.layers.Dense(neurons, activation=actFunc, kernel_initializer=kernelInit, kernel_regularizer=kernelReg, kernel_constraint=kernelConst)
    def __outputLayer(self, name, neurons, actFunc):
        return keras.layers.Dense(neurons, activation=actFunc, name=name)
    def __bnLayer(self):
        return keras.layers.BatchNormalization()
    def __dropoutLayer(self, rate, dropType):
        if(dropType == 'Alpha'): return self.__MCAlphaDropout(rate=rate)
        if(dropType == 'Normal'): return self.__MCDropout(rate=rate)
    
    def build(self, X_train):
        self.model = keras.models.Sequential()
        self.model.add(self.__inputLayer('input', X_train))
        self.model.add(self.__hiddenLayer(4000, 'selu', 'lecun_normal', kernelReg=keras.regularizers.l1(0.0001)))
        self.model.add(self.__hiddenLayer(4000, 'selu', 'lecun_normal', kernelConst=keras.constraints.MaxNorm(1000.)))
        self.model.add(self.__hiddenLayer(2000, 'selu', 'lecun_normal', kernelReg=keras.regularizers.l2(0.0001)))
        self.model.add(self.__hiddenLayer(2000, 'selu', 'lecun_normal', kernelConst=keras.constraints.MaxNorm(1000.)))
        self.model.add(self.__hiddenLayer(1000, 'selu', 'lecun_normal', kernelReg=keras.regularizers.l2(0.0001)))
        self.model.add(self.__hiddenLayer(1000, 'selu', 'lecun_normal', kernelConst=keras.constraints.MaxNorm(1000.)))
        self.model.add(self.__hiddenLayer(500, 'selu', 'lecun_normal', kernelReg=keras.regularizers.l2(0.0001)))
        self.model.add(self.__hiddenLayer(500, 'selu', 'lecun_normal', kernelConst=keras.constraints.MaxNorm(1000.)))
        self.model.add(self.__hiddenLayer(250, 'selu', 'lecun_normal', kernelReg=keras.regularizers.l2(0.0001)))
        self.model.add(self.__hiddenLayer(250, 'selu', 'lecun_normal', kernelConst=keras.constraints.MaxNorm(1000.)))
        self.model.add(self.__hiddenLayer(125, 'selu', 'lecun_normal', kernelReg=keras.regularizers.l2(0.0001)))
        self.model.add(self.__hiddenLayer(125, 'selu', 'lecun_normal', kernelConst=keras.constraints.MaxNorm(1000.)))
        self.model.add(self.__hiddenLayer(60, 'selu', 'lecun_normal', kernelReg=keras.regularizers.l2(0.0001)))
        self.model.add(self.__hiddenLayer(60, 'selu', 'lecun_normal', kernelConst=keras.constraints.MaxNorm(1000.)))
        self.model.add(self.__hiddenLayer(30, 'selu', 'lecun_normal'))
        self.model.add(self.__dropoutLayer(0.25, 'Alpha'))
        self.model.add(self.__hiddenLayer(30, 'selu', 'lecun_normal'))
        self.model.add(self.__dropoutLayer(0.25, 'Alpha'))
        self.model.add(self.__outputLayer('output', 10, 'softmax'))
    
    def get_Info(self, info):
        if(info == "Summary"):
            self.model.summary()
        if(info == "Layer"):
            for i in self.model.layers[1:]:
                weights, biases = i.get_weights()
                print("====================")
                print("Layer Name:", i.name)
                print("Layer Connection Weights Shape:", weights.shape)
                print("Layer Connection Weights:", weights)
                print("Layer Biases Shape:", biases.shape)
                print("Layer Biases:", biases)
                print("====================")
                
    def compile(self, opt, learning_rate=0.01):
        if(opt  == 'ExpDecay'): learning_rate = keras.optimizers.schedules.ExponentialDecay(learning_rate, self.hyperParams['expDecay'][0], self.hyperParams['expDecay'][1])
        if(opt == 'Nesterov'): optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
        if(opt == 'RMSprop'): optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate, rho=0.9)
        if(opt == 'Nadam'): optimizer = keras.optimizers.Nadam(learning_rate=learning_rate)
        self.model.compile(loss=self.hyperParams['loss'], optimizer=optimizer, metrics=self.hyperParams['metrics'])
    
    def fit(self, trainSet, valSet, epochs, batchSize, callBacks=None):
        X_train, Y_train = trainSet
        X_valid, Y_valid = valSet
        self.history = self.model.fit(X_train, Y_train, epochs=epochs, validation_data=(X_valid, Y_valid), batch_size=batchSize, callbacks=callBacks)
    
    def findOptLR(self, trainSet, valSet):
        self.compile(self.hyperParams['optimizer'])
        lr_finder = self.__LRFinder(min_lr=self.hyperParams['lrFinder'][0], max_lr=self.hyperParams['lrFinder'][1])
        self.fit(trainSet, valSet, self.hyperParams['lrFinder'][2], self.hyperParams['batchSize'], [lr_finder])
        self.hyperParams['optLR'] = float(input("\nEnter the observed optimal LR: "))/10.0
        print("Optimal LR is set to:", self.hyperParams['optLR'])
    
    def scheduleLR(self, scheduler):
        self.compile(self.hyperParams['optimizer'], self.hyperParams['optLR'])
        if(scheduler in ('None', 'Exp')): return []
        if(scheduler == 'Perf'): lr_scheduler = keras.callbacks.ReduceLROnPlateau(factor=self.hyperParams['perf'][0], patience=self.hyperParams['perf'][1])
        if(scheduler == '1Cycle'):
            if(self.hyperParams['optimizer'] == 'Nesterov'): lr_scheduler = self.__OneCycleLR(self.hyperParams['optLR'])
            else: lr_scheduler = self.__OneCycleLR(self.hyperParams['optLR'], False)
        return [lr_scheduler]
    
    def plotLearningCurve(self):
        pd.DataFrame(self.history.history).plot()
        plt.grid(True)
        plt.show()
    
    def assemble(self, trainSet, valSet):
        X_train, Y_train = trainSet
        self.build(X_train)
        self.get_Info('Summary')
        
        self.hyperParams = {'optimizer': 'Nesterov',
                            'loss': "sparse_categorical_crossentropy",
                            'metrics': ["accuracy"],
                            'lrFinder': (0.0001, 10.0, 2),
                            'batchSize': 1000,
                            'eStopPat': 10,
                            'scheduler': '1Cycle',
                            'epochs': 15}
        
        self.findOptLR(trainSet, valSet)
        save_best = keras.callbacks.ModelCheckpoint("CIFAR10-NN.h5", save_best_only=True)
        early_stop = keras.callbacks.EarlyStopping(patience=self.hyperParams['eStopPat'], restore_best_weights=True)
        callBacks = [save_best, early_stop] + self.scheduleLR(self.hyperParams['scheduler'])
        self.fit(trainSet, valSet, self.hyperParams['epochs'], self.hyperParams['batchSize'], callBacks)
    
    def evaluate(self, testSet):
        X_test, Y_test = testSet
        self.model.evaluate(X_test, Y_test)
    
    def predict(self, newFeVec, mcDrop=None):
        if(mcDrop is not None):
            preds = np.stack([self.model.predict(newFeVec) for i in range(mcDrop)])
            preds = np.round(((preds).mean(axis=0)),4)
            print("Probabilities:", preds)
        else: preds = self.model.predict(newFeVec)
        return preds.argmax(axis=-1)

cifar = keras.datasets.cifar10
(X_train_full, Y_train_full), (X_test, Y_test) = cifar.load_data()

preProcess = PreProcess()
X_train_full = preProcess.perform(X_train_full)
Y_train_full = Y_train_full.ravel()
print("PreProcessed Attrs: ", X_train_full.shape)
print("PreProcessed Labels: ", Y_train_full.shape)

validationSplit = ValidationSplit(0.2)
X_train, X_valid, Y_train, Y_valid = validationSplit.perform(X_train_full, Y_train_full)
print("Training Attrs: ", X_train.shape)
print("Training Labels: ", Y_train.shape)
print("Validation Attrs: ", X_valid.shape)
print("Validation Labels: ", Y_valid.shape)

neuralNet = NeuralNet()
neuralNet.assemble((X_train, Y_train), (X_valid, Y_valid))
neuralNet.plotLearningCurve()

preds = neuralNet.predict(preProcess.perform(X_test, False))
Y_test = Y_test.ravel()
accuracy = np.round((np.sum(preds == Y_test) / len(Y_test)), 4)
print("Model Accuracy:", accuracy)