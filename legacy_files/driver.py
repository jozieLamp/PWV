import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from statistics import *
import heartpy as hp
import copy
from IPython.display import display
pd.options.display.max_rows = 450

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, Flatten, Dropout, Dense
from tensorflow.compat.v1.keras.layers import CuDNNLSTM   # CuDNNLSTM not yet released for TF 2.0

import pwv
import importlib
importlib.reload(pwv)

waveformData, segmentIndices, plots = pwv.preprocess("/home/anrath/projects/research/josie/PWV/full_demo/Raw Data", returnPlot=[True, True, True])
metrics = pwv.analyzeWave(waveformData, segmentIndices)

metrics = metrics[metrics[0]!= 5]
metrics = metrics[metrics[0]!= 17]

outcomes = [0,1,1,0,1,-1,1,0,0,0,0,0,1,1,1,0,0,-1,0,0,1,0]

metrics['25'] = metrics.apply(lambda row: outcomes[int(row[0])], axis=1)
metrics = metrics.dropna()



vocabulary_size = 10000  # choose 20k most-used words for truncated vocabulary
sequence_length = 1000   # choose 1000-word sequences
embedding_dims = 50      # number of dimensions to represent each word in vector space
batch_size = 32          # feed in the neural network in 100-example training batches
num_epochs = 30          # number of times the neural network goes over EACH training example

x_train, x_test, y_train, y_test = pwv.MLSplit(metrics)
x_test = x_test.to_numpy()

# Set more global constants
num_categories = max(y_train) + 1

# Pad sequences to maximum found sequence length
x_train = pad_sequences(sequences=x_train, maxlen=sequence_length)
# x_test = pad_sequences(sequences=x_test, maxlen=sequence_length)

# Compute batch size and cutoff training & validation examples to fit
training_cutoff, test_cutoff = len(x_train) % batch_size, len(x_test) % batch_size
x_train, y_train = x_train[:-training_cutoff], y_train[:-training_cutoff]
x_test, y_test = x_test[:-test_cutoff], y_test[:-test_cutoff]


# Create word-level multi-class document classification model
# Input Layer
x = Input(shape=(sequence_length,), batch_size=batch_size)

# Word-Embedding Layer
embedded = Embedding(input_dim=vocabulary_size, output_dim=embedding_dims)(x)

# Recurrent Layers
encoder_output = CuDNNLSTM(units=128)(embedded)

# Prediction Layer
y = Dense(units=num_categories, activation='softmax')(encoder_output)

# Compile model
model = Model(inputs=x, outputs=y)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Train multi-class classification model
model.fit(x=x_train, y=y_train,
          validation_data=(x_test, y_test),
          epochs=num_epochs, batch_size=batch_size)