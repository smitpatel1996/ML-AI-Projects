import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

class Prepare():
    def encode(self, data, training=True):
        if(training):
            self.tokenizer = keras.preprocessing.text.Tokenizer(char_level=True)
            self.tokenizer.fit_on_texts(data)
        encoded = (np.array(self.tokenizer.texts_to_sequences([data]))-1)[0]
        return encoded, len(self.tokenizer.word_index)
    
    def oneHot(self, attrs):
        max_id = len(self.tokenizer.word_index)
        return keras.utils.to_categorical(attrs, num_classes=max_id)
    
    def getChar(self, encode):
        return self.tokenizer.sequences_to_texts(encode+1)

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
    def __RNNLayer(self, neurons, dp, recDp, cell="GRU"):
        if(cell == "GRU"):
            return keras.layers.GRU(neurons, return_sequences=True, dropout=dp, recurrent_dropout=recDp)
        if(cell == "LSTM"):
            return keras.layers.LSTM(neurons, return_sequences=True, dropout=dp, recurrent_dropout=recDp)
    def __TDLayer(self, neurons, activation):
        return keras.layers.TimeDistributed(keras.layers.Dense(neurons, activation))
    
    def build(self, X_train):
        self.model = keras.models.Sequential([
            keras.layers.GRU(256, return_sequences=True, input_shape=[None, X_train.shape[2]], dropout=0.4, recurrent_dropout=0.4),
            keras.layers.GRU(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
            keras.layers.GRU(64, return_sequences=True, dropout=0.1, recurrent_dropout=0.1),
            keras.layers.TimeDistributed(keras.layers.Dense(X_train.shape[2], activation="softmax"))
        ])
        
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
                            'lrFinder': (0.0001, 1000.0, 5),
                            'batchSize': 100,
                            'eStopPat': 15,
                            'scheduler': '1Cycle',
                            'epochs': 50}
        
        self.findOptLR(trainSet, valSet)
        save_best = keras.callbacks.ModelCheckpoint("GenText-RNN.h5", save_best_only=True)
        early_stop = keras.callbacks.EarlyStopping(patience=self.hyperParams['eStopPat'], restore_best_weights=True)
        callBacks = [save_best, early_stop] + self.scheduleLR(self.hyperParams['scheduler'])
        self.fit(trainSet, valSet, self.hyperParams['epochs'], self.hyperParams['batchSize'], callBacks)
    
    def loadModel(self, modelFile):
        self.model = keras.models.load_model(modelFile)
    
    def evaluate(self, testSet):
        X_test, Y_test = testSet
        self.model.evaluate(X_test, Y_test)
    
    def predict(self, newFeVec, mcDrop=None):
        if(mcDrop is not None):
            preds = np.stack([self.model.predict(newFeVec) for i in range(mcDrop)])
            preds = np.round(((preds).mean(axis=0)),4)
            print("Probabilities:", preds)
        else: preds = self.model.predict(newFeVec)
        return preds

### ==== ACTUAL IMPLEMENTATION ==== ###

def splitSeq(attrs, labels):
    dsize = len(attrs)
    trainLen = int(dsize*0.8)
    valLen = int(dsize*0.2)
    X_train = attrs[:trainLen]
    Y_train = labels[:trainLen]
    X_val = attrs[trainLen:trainLen+valLen]
    Y_val = labels[trainLen:trainLen+valLen]
    return (X_train, Y_train), (X_val, Y_val)

def generateAttrsLabels(data, inTS, fTS):
    l = len(data)
    X = np.empty((0,inTS), float)
    Y = np.empty((0,inTS,fTS), float)
    for i in range(inTS, (l-fTS)+1, 1):
        X = np.append(X, np.array([data[i-inTS:i]]), axis=0)
        subY = np.empty((0,fTS), float)
        for j in range(i-inTS, i):
            subY = np.append(subY, np.array([data[j+1:j+fTS+1]]), axis=0)
        Y = np.append(Y, np.array([subY]), axis=0)
    return X, Y

f = open("shakespeare.txt", "r")
book = f.read()
f.close()

prepare = Prepare()
encoded, max_id = prepare.encode(book[:20000])

inputTimeSteps = 25
forecastTimeSteps = 1
X, Y = generateAttrsLabels(encoded, inputTimeSteps, forecastTimeSteps)
X = prepare.oneHot(X)

(X_train, Y_train), (X_val, Y_val) = splitSeq(X, Y)
print("Training Attrs: ", X_train.shape)
print("Training Labels: ", Y_train.shape)
print("Validation Attrs: ", X_val.shape)
print("Validation Labels: ", Y_val.shape)

neuralNet = NeuralNet()
neuralNet.assemble((X_train, Y_train), (X_val, Y_val))
neuralNet.plotLearningCurve()
#neuralNet.loadModel('GenText-RNN.h5')

def next_char(text, temperature=1):
    encoded, _ = prepare.encode(text, training=False)
    X_new = prepare.oneHot(encoded)
    X_new = X_new.reshape(1, X_new.shape[0], X_new.shape[1])
    y_proba = neuralNet.predict(X_new)[0, -1:, :][0]
    idx = (-y_proba).argsort()[int(len(y_proba)*0.20):]
    y_proba[idx] = 0
    y_proba = y_proba/np.sum(y_proba)
    out = np.array(range(len(y_proba)))
    a = np.random.choice(out, p=y_proba)
    return (prepare.getChar(np.array([[a]])))[0]
    
def complete_text(text, n_chars=5, temperature=1):
    for _ in range(n_chars):
        text += next_char(text, temperature) 
    return text

string = "my name i"
l = complete_text(string, 10) 
print(l)