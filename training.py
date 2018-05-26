import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import keras
from keras import optimizers
from keras import backend as K
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.utils import plot_model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
In [ ]:
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import os, re, csv, math, codecs

sns.set_style("whitegrid")
np.random.seed(0)

DATA_PATH = '../input/'
EMBEDDING_PATH = '../input/'

MAX_NB_WORDS = 100000
tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))
stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
In [ ]:
# Load Embedding
print('load word embedding...')
embedding_index = {}
f = codecs.open('../input/fasttext/wiki.simple1.vec', encoding='utf-8')
for line in tqdm(f):
    values = line.rstrip().rsplit(' ')
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embedding_index[word] = coefs
f.close()
print('found %s word vectors' % len(embedding_index))
In [ ]:
# Load Data
input_dir = '../input/jigsaw-toxic-comment-classification-challenge/'
train_df = pd.read_csv(input_dir + 'train.csv', sep=',', header=0)
test_df = pd.read_csv(input_dir + 'test.csv', sep=',', header=0)

test_df = test_df.fillna('_NA_')
print('Training DF: '+str(train_df.shape))
print('Testing DF: '+str(test_df.shape))

label_names = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y_train = train_df[label_names].values

print(y_train.shape)

# Visualize
train_df['doc_len'] = train_df['comment_text'].apply(lambda words: len(words.split(" ")))
max_seq_len = np.round(train_df['doc_len'].mean() + train_df['doc_len'].std()).astype(int)
sns.distplot(train_df['doc_len'], hist=True, kde=True, color='b', label='doc_len')
plt.axvline(x=max_seq_len, color='k', linestyle='--', label='max len')
plt.title('comment length');
plt.legend()
plt.show()
In [ ]:
raw_docs_train = train_df['comment_text'].tolist()
raw_docs_test = test_df['comment_text'].tolist()
num_classes = len(label_names)

print('pre-processing training data')
processed_train_data = []
for doc in tqdm(raw_docs_train):
    tokens = tokenizer.tokenize(doc)
    filtered = [word for word in tokens if word not in stop_words]
    processed_train_data.append(" ".join(filtered))

print('pre-processing test data')
processed_test_data = []
for doc in tqdm(raw_docs_test):
    tokens = tokenizer.tokenize(doc)
    filtered = [word for word in tokens if word not in stop_words]
    processed_test_data.append(" ".join(filtered))

print("Tokenizing input data....")
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=True, char_level=False)
tokenizer.fit_on_texts(processed_train_data + processed_test_data)
word_seq_train = tokenizer.texts_to_sequences(processed_train_data)
word_seq_test = tokenizer.texts_to_sequences(processed_test_data)
word_index = tokenizer.word_index
print("Dictionary Size: "+str(len(word_index)))

# Pad Sequences
word_seq_train = sequence.pad_sequences(word_seq_train, maxlen=max_seq_len)
word_seq_test = sequence.pad_sequences(word_seq_test, maxlen=max_seq_len)
In [ ]:
# Training Parameters
batch_size = 256
num_epochs = 8

num_filters = 64
embed_dim = 300
weight_decay = 1e-4
In [ ]:
# Embedding Matrix
words_not_found = []
nb_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_dim))

for word, i in word_index.items():
    if(i>=nb_words):
        continue
    embedding_vector = embedding_index.get(word)
    if(embedding_vector is not None) and len(embedding_vector)>0:
        embedding_matrix[i] = embedding_vector
    else:
        words_not_found.append(word)
print('Number of null words embeddings found %d' % np.sum(np.sum(embedding_matrix, axis=1)==0))
In [ ]:
print(' Sample words not found: ',np.random.choice(words_not_found, 10))
In [ ]:
print('Building CNN')
model = Sequential()
model.add(Embedding(nb_words, embed_dim, weights=[embedding_matrix], input_length=max_seq_len, trainable=False))
model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))
model.add(MaxPooling1D(2))
model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))
model.add(GlobalMaxPooling1D())
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Dense(num_classes, activation='sigmoid'))

adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
model.summary()
In [ ]:
# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=4, verbose=1)
callback_list = [early_stopping]
In [ ]:
# Training CNN
hist = model.fit(word_seq_train, y_train, batch_size=batch_size, epochs=num_epochs, callbacks=callback_list, validation_split=0.1, shuffle=True, verbose=2)
In [ ]:
y_test = model.predict(word_seq_test)
In [ ]:
#create a submission
submission_df = pd.DataFrame(columns=['id'] + label_names)
submission_df['id'] = test_df['id'].values 
submission_df[label_names] = y_test 
submission_df.to_csv("./submit.csv", index=False)
In [ ]:
# Generate Plots
plt.figure()
plt.plot(hist.history['loss'], lw=2.0, color='b', label='train')
plt.plot(hist.history['val_loss'], lw=2.0, color='r', label='val')
plt.title('CNN Toxic Sentiment')
plt.xlabel('Epochs')
plt.ylabel('Binary Cross-Entropy Loss')
plt.legend(loc='upper right')
plt.show()
In [ ]:
plt.figure()
plt.plot(hist.history['acc'], lw=2.0, color='b', label='train')
plt.plot(hist.history['val_acc'], lw=2.0, color='r', label='val')
plt.title('CNN Toxic Sentiment')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
plt.show()
