# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 11:29:52 2022

@author: andre
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, TimeDistributed,Dropout,Input, Dense,\
    BatchNormalization, GRU, Layer, Flatten
from tensorflow.keras.regularizers import l2
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

# Load in Left side .WAV Data.
X2 = [] 
count2 = 0
database_path2 = "C:\\Users\\andre\\OneDrive\\Documents\\ESI2022\\MLDatabases\\Right\\"
for filename2 in glob.glob(os.path.join(database_path2, '*.wav')):
    X2.append(wavfile.read(filename2)[1])
    count2 = count2 + 1    

# Get the smallest size audio file (this will be sample size input to model)
sample_size = len(X1[0])
for data in X1:
    if len(data) < sample_size:
        sample_size = len(data)

# Make audio data into equal size chunks
X1e = []
for i in X1:
    num_chunks = len(i)//sample_size
    for j in range(num_chunks):
        X1e.append(i[(j+1)*sample_size-sample_size:(j+1)*sample_size])
X1 = X1e
        
X2e = []
for i in X2:
    num_chunks = len(i)//sample_size
    for j in range(num_chunks):
        X2e.append(i[(j+1)*sample_size-sample_size:(j+1)*sample_size])        
X2=X2e  

del X1e
del X2e   

# Create Output data that is the same length as the input data.
Y1 = np.ones([X1.__len__()],dtype='float32').tolist()
Y2 = np.zeros([X2.__len__()],dtype='float32').tolist()


# Concatenate Left and Right .WAV data and output data as numpy arrays.
X1.extend(X2)
X = np.asarray(X1)
Y = np.asarray(Y1+Y2).astype(np.int16)

#X=list(X)
#Y=list(Y)

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

def nn(shape_1,shape_2):
    input = Input(shape=[shape_1,shape_2,2,1])

    conv1 = TimeDistributed(Conv2D(filters=32, kernel_size=[32,1], activation='relu',strides =(3,1)))(input)
    batch1 = TimeDistributed(BatchNormalization())(conv1)

    conv2 = TimeDistributed(Conv2D(filters=32, kernel_size=[32,1], activation='relu',strides =(2,1)))(batch1)
    batch2 = TimeDistributed(BatchNormalization())(conv2)

    conv3 = TimeDistributed(Conv2D(filters=32, kernel_size=[32,1], activation='relu',strides =(2,1)))(batch2)
    batch3 = TimeDistributed(BatchNormalization())(conv3)

    conv4 = TimeDistributed(Conv2D(filters=32, kernel_size=[32,1], activation='relu',strides =(2,1)))(batch3)
    batch4 = TimeDistributed(BatchNormalization())(conv4)

    flat = TimeDistributed(Flatten())(batch4)


    gru1 = GRU(256, activation='relu',return_sequences=True, kernel_regularizer=l2(0.01))(flat)
    drop1 = Dropout(rate=0.4)(gru1)
    batch1 = BatchNormalization()(drop1)

    gru2 = GRU(128, activation='relu',return_sequences=True, kernel_regularizer=l2(0.01))(batch1)
    drop2 = Dropout(rate=0.4)(gru2)
    batch2 = BatchNormalization()(drop2)


    dense = TimeDistributed(Dense(2, activation='softmax'),name = 'output')(batch2)


    return [input], [dense]

# Define Training Parameters
EPOCH_LENGTH = 30
SAMPLE_RATE = 44100
NUM_BATCH_SIZE = 1

input, output = nn(1,sample_size)
model = Model(inputs=input,outputs=output)

optimizer = Adam(learning_rate=2*1e-4)

# Compile Model
model.compile(optimizer=optimizer, loss={
                  'output': 'sparse_categorical_crossentropy', },
              metrics={
                  'output': 'sparse_categorical_accuracy', },
              sample_weight_mode='temporal')
model.summary()


# Save the most accurate model to file. (Verbosity Gives more information)
checkpointer = ModelCheckpoint(filepath="SavedModels/checkpointModel.hdf5", verbose=1,save_best_only=True)

# Start the timer
start = datetime.now()

# Train the model
model.fit(X_train,Y_train,batch_size=NUM_BATCH_SIZE, epochs=EPOCH_LENGTH, validation_data=(X_test,Y_test), callbacks=[checkpointer],verbose=1)

# Get and Print Model Validation Accuracy
test_accuracy=model.evaluate(X_test,Y_test,verbose=0)
print(test_accuracy[1])
