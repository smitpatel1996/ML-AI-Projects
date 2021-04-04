import time 
import regex as re
import numpy as np
import pandas as pd
import tensorflow as tf
from string import digits
from tensorflow import keras
from operator import itemgetter
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

class ValidationSplit():
    def __init__(self, val_size):
        self.val_size = val_size
    
    def perform(self, attrs, labels):
        index = np.random.choice(attrs.shape[0], int(len(attrs)*(1-self.val_size)), replace=False)
        return (np.take(attrs, index, axis=0), np.delete(attrs, index, axis=0), np.take(labels, index, axis=0), np.delete(labels, index, axis=0))

class Enhance(): 
    def perform(self, npInp, training=True):
        self.max_length = 20
        if(training):
            size_of_vocab = 50000
            self.tokenizer = keras.preprocessing.text.Tokenizer(oov_token="<OOV>")
            self.tokenizer.fit_on_texts(npInp)
            self.tokenizer.word_index = dict(sorted(self.tokenizer.word_index.items(), key = itemgetter(1))[:size_of_vocab])
            self.vocab_size = len(self.tokenizer.word_index) + 1
        encoded = self.tokenizer.texts_to_sequences(npInp)
        paddedVecs = keras.preprocessing.sequence.pad_sequences(encoded, maxlen=self.max_length, padding='post', truncating='post')
        return paddedVecs
    
    def getEmbeddingParams(self):
        return (self.vocab_size, self.max_length, self.tokenizer)
    
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
        
    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state = hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
    
    def call(self, query, values):
        hidden_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(self.dec_units)
        
    def call(self, x, hidden, enc_output):
        context_vector = self.attention(hidden, enc_output)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)
        return x, state

class CustomNet():
    @tf.function
    def __train_step(self, inp, targ, enc_hidden):
        loss = 0
        with tf.GradientTape() as tape:
            enc_output, enc_hidden = self.encoder(inp, enc_hidden)
            dec_hidden = enc_hidden
            dec_input = tf.expand_dims([self.tar_tokenizer.word_index['sos']] * self.BATCH_SIZE, 1)
            for t in range(1, targ.shape[1]):
                predictions, dec_hidden = self.decoder(dec_input, dec_hidden, enc_output)
                loss += self.__loss_function(targ[:, t], predictions)
                dec_input = tf.expand_dims(targ[:, t], 1)
        batch_loss = (loss / int(targ.shape[1]))
        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return batch_loss
    
    def __loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)

    def __init__(self, trainSet, embeddingParams):
        X_train, Y_train = trainSet
        sourceParams, targetParams = embeddingParams
        BUFFER_SIZE, self.BATCH_SIZE = (len(X_train), 64)
        self.steps_per_epoch = BUFFER_SIZE//self.BATCH_SIZE
        vocab_inp_size, vocab_tar_size = (sourceParams[0], targetParams[0])
        self.inp_tokenizer, self.tar_tokenizer = (sourceParams[2], targetParams[2])
        self.max_length = sourceParams[1]
        self.dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).shuffle(BUFFER_SIZE)
        self.dataset = self.dataset.batch(self.BATCH_SIZE, drop_remainder=True)
        embedding_dim, self.units = (64, 128)
        self.encoder = Encoder(vocab_inp_size, embedding_dim, self.units, self.BATCH_SIZE)
        self.decoder = Decoder(vocab_tar_size, embedding_dim, self.units, self.BATCH_SIZE)
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    def train(self):
        EPOCHS = 2
        for epoch in range(EPOCHS):
            start = time.time()
            enc_hidden = self.encoder.initialize_hidden_state()
            total_loss = 0
            for (batch, (inp, targ)) in enumerate(self.dataset.take(self.steps_per_epoch)):
                batch_loss = self.__train_step(inp, targ, enc_hidden)
                total_loss += batch_loss
                if batch % 2 == 0:
                    print('Epoch {} Batch {} loss {}'.format(epoch + 1,batch, batch_loss.numpy()))
            print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / self.steps_per_epoch))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
    
    def translate(self, inputs):
        inputs = tf.convert_to_tensor(inputs)
        result = ''
        hidden = [tf.zeros((1, self.units))]
        enc_out, enc_hidden = self.encoder(inputs, hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([self.tar_tokenizer.word_index['sos']], 0)
        for t in range(self.max_length):
            predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)
            attention_weights = tf.reshape(attention_weights, (-1, ))
            predicted_id = tf.argmax(predictions[0]).numpy()
            result += self.tar_tokenizer.index_word[predicted_id] + ' '
            if self.tar_tokenizer.index_word[predicted_id] == 'eos':
                return result
            dec_input = tf.expand_dims([predicted_id], 0)
        return result

### ==== ACTUAL IMPLEMENTATION ==== ###
lines_raw = pd.read_csv('spa.txt', sep='\t', header=None, usecols=[0, 1])
lines_raw = lines_raw.head(100000)
lines_raw.columns = ['source', 'target']

labels = ['target']
split = Split(0.2, labels)
trainDF, testDF, Y_train_full, Y_test = split.perform(lines_raw)
X_train_full = trainDF.to_numpy().flatten()
X_test = testDF.to_numpy().flatten()

validationSplit = ValidationSplit(0.1)
X_train, X_valid, Y_train, Y_valid = validationSplit.perform(X_train_full, Y_train_full)

def refine(sentence):
    num_digits = str.maketrans('','', digits)
    sentence = sentence.lower()
    sentence = re.sub(" +", " ", sentence)
    sentence = sentence.translate(num_digits)
    sentence = re.sub(r"([?.!,¿])", r"\1", sentence)
    sentence = sentence.rstrip().strip()
    sentence =  'sos ' + sentence + ' eos'
    return sentence

print(X_valid[11])
print(X_valid[26])

X_train = np.array(list(map(lambda x: refine(x), X_train)))
Y_train = np.array(list(map(lambda x: refine(x), Y_train)))
X_valid = np.array(list(map(lambda x: refine(x), X_valid)))
Y_valid = np.array(list(map(lambda x: refine(x), Y_valid)))
X_test = np.array(list(map(lambda x: refine(x), X_test)))
Y_test = np.array(list(map(lambda x: refine(x), Y_test)))

enhanceSource = Enhance()
X_train = enhanceSource.perform(X_train)
X_valid = enhanceSource.perform(X_valid, False)
enhanceTarget = Enhance()
Y_train = enhanceTarget.perform(Y_train)
Y_valid = enhanceTarget.perform(Y_valid, False)

print("Training Attrs: ", X_train.shape)
print("Training Labels: ", Y_train.shape)
print("Validation Attrs: ", X_valid.shape)
print("Validation Labels: ", Y_valid.shape)

customNet = CustomNet((X_train, Y_train), (enhanceSource.getEmbeddingParams(), enhanceTarget.getEmbeddingParams()))
customNet.train()
customNet.translate(X_valid[11])
customNet.translate(X_valid[26])
print(Y_valid[11])
print(Y_valid[26])