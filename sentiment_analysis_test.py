from keras.datasets import imdb
import sys
from keras.preprocessing import sequence
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.models import load_model
import numpy as np

vocabulary_size = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = vocabulary_size)
print('Loaded dataset with {} training samples, {} test samples'.format(len(X_train), len(X_test)))

print('---review---')
print(X_train[6])
print('---label---')
print(y_train[6])


word2id = imdb.get_word_index()
id2word = {i: word for word, i in word2id.items()}
print('---review with words---')
print([id2word.get(i, ' ') for i in X_train[6]])
print('---label---')
print(y_train[6])


print('Maximum review length: {}'.format(
len(max((X_train + X_test), key=len))))


print('Minimum review length: {}'.format(
len(min((X_test + X_test), key=len))))


max_words = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)

cd = np.array([X_train[37]])
print(cd.shape)

model = load_model('tmp.h5')

print(model.predict(cd))
print(y_train[37])
# scores = model.evaluate(X_test, y_test, verbose=0)
# model.evaluate(X_test[0], y=None, batch_size=None, verbose=1)
# print(X_train[0].shape)