
# coding: utf-8

# In[*]


import numpy as np
np.random.seed(42)
import pandas as pd


# In[*]


from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


# In[*]


from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.preprocessing import text, sequence
from keras.callbacks import Callback


# In[*]


import warnings
warnings.filterwarnings('ignore')


# In[*]


import os
os.environ['OMP_NUM_THREADS'] = '4'


# In[*]


EMBEDDING_FILE = 'data/fasttext_300d_crawl_2m/crawl-300d-2M.vec'


# In[*]


train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')


# In[*]


X_train = train["comment_text"].values
y_train = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
X_test = test["comment_text"].values


# In[*]


max_features = 30000
maxlen = 100
embed_size = 300


# In[*]


tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train) + list(X_test))
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
x_train = sequence.pad_sequences(X_train, maxlen=maxlen)
x_test = sequence.pad_sequences(X_test, maxlen=maxlen)


# In[*]


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


# In[*]


embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE))


# In[*]


word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


# In[*]


class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super().__init__()
        self.interval = interval
        self.X_val, self.y_val = varslidation_data
        
    def on_epoch_end(self, epoch, logs={}):
        if epock % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print('\n ROC_AUC - epoch: {} - score: {:0.6f} \n'.format(epoch+1, score))


# In[*]


def get_model():
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(GRU(80, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    outp = Dense(6, activation='sigmoid')(conc)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

model = get_model()


# In[*]


batch_size = 32
epochs = 2


# In[*]


X_train, X_validation, y_train, y_validation = train_test_split(x_train, y_train, train_size=0.95, random_state=233)
RocAuc = RocAucEvaluation(validation_data=(X_validation, y_validation), interval=1)


# In[*]


hist = model.fit(
    X_train, y_train, batch_size=batch_size, epochs=epochs, 
    validation_data=(X_validation, y_validation), callbacks=[RocAuc], verbose=2)


# In[*]


y_pred = model.predict(x_test, batch_size=1024)

