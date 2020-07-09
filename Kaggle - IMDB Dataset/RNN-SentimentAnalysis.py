import os
import regex as re
import sklearn
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from operator import itemgetter
import matplotlib.pyplot as plt
from gensim.parsing.preprocessing import remove_stopwords as rmvStop
from sklearn.model_selection import train_test_split as tts


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
            return self.__get_tts_from_df(dataFrame, self.test_size, attrs, self.labels, dataFrame[self.labels])
        else:
            return self.__get_tts_from_df(dataFrame, self.test_size, attrs, self.labels)

class BinarizeLabels():
    def perform(self, labels, training=True):
        if(training):
            self.lb = sklearn.preprocessing.LabelBinarizer()
            self.lb.fit(labels)
        return self.lb.transform(labels)
    
class Enhance(): 
    def perform(self, npInp, training=True):
        self.max_length = 250
        if(training):
            size_of_vocab = 5000
            self.tokenizer = keras.preprocessing.text.Tokenizer(num_words=size_of_vocab, oov_token="<OOV>")
            self.tokenizer.fit_on_texts(npInp)
            self.tokenizer.word_index = dict(sorted(self.tokenizer.word_index.items(), key = itemgetter(1))[:size_of_vocab]) 
            self.vocab_size = len(self.tokenizer.word_index) + 1
        encoded = self.tokenizer.texts_to_sequences(npInp)
        paddedVecs = keras.preprocessing.sequence.pad_sequences(encoded, maxlen=self.max_length, padding='post', truncating='post')
        return paddedVecs
    
    def getEmbeddingParams(self):
        return (self.vocab_size, self.max_length, self.tokenizer)

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
    
    def __GloveEmbeddings(self, filename, embeddingParams, output_dim):
        embeddings_index = dict()
        f = open(filename)
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        embedding_matrix = np.zeros((embeddingParams[0], output_dim))
        for word, i in embeddingParams[2].word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        return embedding_matrix    
    def __denseLayer(self, neurons, actFunc, kernelInit=None, kernelReg=None, kernelConst=None):
        return keras.layers.Dense(neurons, activation=actFunc, kernel_initializer=kernelInit, kernel_regularizer=kernelReg, kernel_constraint=kernelConst)
    def __EmbeddingLayer(self, embeddingParams, output_dim, preTrained=False):
        if(preTrained):
            embedding_matrix = self.__GloveEmbeddings('glove.6B.50d.txt', embeddingParams, output_dim)
            return keras.layers.Embedding(embeddingParams[0], output_dim, input_length=embeddingParams[1], mask_zero=True, weights=[embedding_matrix], trainable=False)
        return keras.layers.Embedding(embeddingParams[0], output_dim, input_length=embeddingParams[1], mask_zero=True)
    def __RNNLayer(self, neurons, dp, recDp, returnSeq=True, cell="GRU"):
        if(cell == "GRU"):
            return keras.layers.GRU(neurons, return_sequences=returnSeq, dropout=dp, recurrent_dropout=recDp)
        if(cell == "LSTM"):
            return keras.layers.LSTM(neurons, return_sequences=returnSeq, dropout=dp, recurrent_dropout=recDp)
    def __TDLayer(self, neurons, activation):
        return keras.layers.TimeDistributed(keras.layers.Dense(neurons, activation))
    
    def build(self, X_train, embeddingParams):
        self.model = keras.models.Sequential()
        self.model.add(self.__EmbeddingLayer(embeddingParams, 50, True))
        self.model.add(self.__RNNLayer(64, 0.2, 0.2))
        self.model.add(self.__RNNLayer(32, 0.15, 0.15))
        self.model.add(self.__RNNLayer(16, 0.1, 0.1, False))
        self.model.add(self.__denseLayer(1, 'sigmoid'))
               
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
    
    def assemble(self, trainSet, valSet, embeddingParams):
        X_train, Y_train = trainSet
        self.build(X_train, embeddingParams)
        self.get_Info('Summary')
        
        self.hyperParams = {'optimizer': 'Nesterov',
                            'loss': "binary_crossentropy",
                            'metrics': ["accuracy"],
                            'lrFinder': (0.0001, 1000.0, 5),
                            'batchSize': 500,
                            'eStopPat': 10,
                            'scheduler': '1Cycle',
                            'epochs': 50}
        
        self.findOptLR(trainSet, valSet)
        save_best = keras.callbacks.ModelCheckpoint("SentimentAnalysis-RNN.h5", save_best_only=True)
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
reviews = pd.read_csv("IMDB_Dataset.csv", sep=",")

labels = ['sentiment']
split = Split(0.1, labels, True, labels)
trainDF, testDF, Y_train_full, Y_test = split.perform(reviews)

binarize = BinarizeLabels()
X_train_full = trainDF.to_numpy().flatten()
X_test = testDF.to_numpy().flatten()
Y_train_full = binarize.perform(Y_train_full).flatten()
Y_test = binarize.perform(Y_test, False).flatten()

validationSplit = ValidationSplit(0.2)
X_train, X_valid, Y_train, Y_valid = validationSplit.perform(X_train_full, Y_train_full)

TAG_RE = re.compile(r'<[^>]+>')
def remove_tags(text):
    return TAG_RE.sub('', text)

def firstWords(inp):
    sentence = remove_tags(inp)
    sentence = re.sub(r'https:\/\/[a-zA-Z]*\.com',' ',sentence)
    sentence = re.sub(r'\s+',' ',sentence)
    sentence = re.sub(r"\b[a-zA-Z]\b", ' ', sentence)
    sentence = re.sub(r'\W+',' ',sentence)
    sentence = sentence.lower()
    sentence = rmvStop(sentence)
    return sentence

X_train = np.array(list(map(lambda x: firstWords(x), X_train)))
X_valid = np.array(list(map(lambda x: firstWords(x), X_valid)))
X_test = np.array(list(map(lambda x: firstWords(x), X_test)))

enhance = Enhance()
X_train = enhance.perform(X_train)
X_valid = enhance.perform(X_valid, False)

print("Training Attrs: ", X_train.shape)
print("Training Labels: ", Y_train.shape)
print("Validation Attrs: ", X_valid.shape)
print("Validation Labels: ", Y_valid.shape)

neuralNet = NeuralNet()
neuralNet.assemble((X_train, Y_train), (X_valid, Y_valid), enhance.getEmbeddingParams())
neuralNet.plotLearningCurve()