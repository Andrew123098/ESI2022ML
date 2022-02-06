# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 15:51:56 2022

@author: andre
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn import metrics
from scipy.io import wavfile
import os
import glob


# Load in Right Side .WAV Data.
X1 = []
count1 = 0
database_path = "C:\\Users\\andre\\OneDrive\\Documents\\ESI2022\\MLDatabases\\Right\\"
for filename in glob.glob(os.path.join(database_path, '*.wav')):
    X1.append(wavfile.read(filename)[1])
    count1 = count1 + 1

# Create Output data that is the same length as the input data.
Y1 = np.ones([X1.__len__()],dtype='float32').tolist()

# Load in Left side .WAV Data.
X2 = [] 
count2 = 0
database_path2 = "C:\\Users\\andre\\OneDrive\\Documents\\ESI2022\\MLDatabases\\Right\\"
for filename2 in glob.glob(os.path.join(database_path2, '*.wav')):
    X2.append(wavfile.read(filename2)[1])
    count2 = count2 + 1    

# Create Output data that is the same length as the input data.
Y2 = np.zeros([X2.__len__()],dtype='float32').tolist()

# Concatenate Left and Right .WAV data and output data as numpy arrays.
X = np.asarray(X1+X2)
Y = np.asarray(Y1+Y2).astype(np.float32)

X = X.tolist()

# Split data into test training data.
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0,shuffle=True)

'''  
print(X[1])    
time = np.linspace(0.,33792, 33792)
plt.plot(time, X[1][:,1], label="Left channel")
plt.plot(time, X[1][:,0], label="Right channel")
plt.legend()
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.show()
'''

# Create the Model
model = Sequential()
'''
### FIRST LAYER
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))

### SECOND LAYER
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dropout(0.5))

### THIRD LAYER
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))

### FINAL LAYER
model.add(Dense(50))
model.add(Activation('softmax'))
'''
# Add an Embedding layer expecting input vocab of size 1000, and
# output embedding dimension of size 64.
model.add(layers.LSTM((1),batch_input_shape=(len(X_train),None,1),return_sequences=True))
model.add(layers.LSTM((1),return_sequences=False))
# Add a LSTM layer with 128 internal units.
#model.add(layers.LSTM(128))

# Add a Dense layer with 10 units.
#model.add(layers.Dense(10))

# Compile Model
history = model.compile(loss='mean_absolute_error', metrics=['accuracy'],optimizer='adam')

model.summary()

# Define Training Parameters
num_epochs = 200
num_batch_size = 32

# Save the most accurate model to file. (Verbosity Gives more information)
checkpointer = ModelCheckpoint(filepath="SavedModels/checkpointModel.hdf5", verbose=1,save_best_only=True)

# Start the timer
start = datetime.now()

# Train the model
model.fit(X_train,Y_train,batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_test,Y_test), callbacks=[checkpointer],verbose=1)

# Get and Print Model Validation Accuracy
test_accuracy=model.evaluate(X_test,Y_test,verbose=0)
print(test_accuracy[1])

