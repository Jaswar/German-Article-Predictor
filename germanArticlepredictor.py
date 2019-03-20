# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 18:27:03 2019

@author: janwa
"""

import pandas as pd
import numpy as np
import sys
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from keras.optimizers import Adam

maxWordLength = 30                              #the maximum length of a word, this will also create a neural net with input shape of this multiplied by alphabet's length see README.md for more info or contact me
convLetters = 4                                 #how long we want our convolutional neural network feature should be - 4 means that every feature will be 4 letters long, which means it will be the size of this multiplied by alphabet's length
filepathToSave = 'convGPUModel.h5'              #filepath to save our trained model
filepathToOpen = 'articlePredictorConv2.h5'     #filepath to open a pretrained model
train = False                                   #if this is set to True we train our model if this is False then we test our pretrained model
learningRate = 0.001                            #the learning rate of a model
batchSize = 128                                 #the batch size our neural net is trained on 
nbEpochs = 20                                   #how many epochs we want to train our model



alphabet = 'abcdefghijklmnopqrstuvwxyzßäöü'
inputSize = len(alphabet) * maxWordLength
dataset = pd.read_csv('wordsAll.txt', delimiter = '\t', header = None,encoding = 'latin-1')



class Predictor(object):
        def __init__(self, nI = 100, nO = 3, lr = 0.001):
            self.learningRate = lr
            self.numInputs = nI
            self.numOutputs = nO
            self.model = Sequential()
            self.model.add(Conv1D(128, int(convLetters*len(alphabet)), input_shape = (self.numInputs, 1), activation = 'relu'))
            self.model.add(MaxPooling1D(pool_size = 2))
            self.model.add(Conv1D(64, int(convLetters*len(alphabet)/2), activation = 'relu'))
            self.model.add(MaxPooling1D(pool_size = 2))
            self.model.add(Flatten())
            self.model.add(Dense(units = 256, activation = 'sigmoid', input_shape = (self.numInputs, )))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(units = 64, activation = 'sigmoid'))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(units = 8, activation = 'sigmoid'))
            self.model.add(Dropout(0.1))
            self.model.add(Dense(units = self.numOutputs, activation = 'softmax'))
            self.model.compile(loss = 'categorical_crossentropy', optimizer = Adam(lr = self.learningRate), metrics = ['accuracy'])
        
        def loadModel(self, filepath):
            self.model = load_model(filepath)
            return self.model
        
predictor = Predictor(nI = inputSize, nO = 3, lr = learningRate)        

if train:
    
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())
    
    articles = dataset.iloc[:,1]
    oheArticles = np.zeros((len(articles), 3))
    
    for i in range(len(articles)):
        oheArticles[i] =  [int(g) for g in articles[i].split(',')]
        argMax = np.argmax(oheArticles[i])
        oheArticles[i] = 0
        oheArticles[i][argMax] = 1
        
    oheWords = np.zeros((dataset.shape[0], inputSize))
    
    for w in range(dataset.shape[0]):
        word = str(dataset[0][w])
        print('Converting: ' + str(word))
        for l in range(len(word)):
            letter = word[l]
            try:
                inx = alphabet.index(letter)
            except:
                sys.exit('This character is not a part of alphabet: ' + letter)
            oheWords[w][l*len(alphabet) + inx] = 1
          
    oheWords = oheWords.reshape((oheWords.shape[0],oheWords.shape[1]),1)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(oheWords, oheArticles, test_size = 0.2, random_state = 0)
    
    
    model = predictor.model
    model.fit(X_train, y_train, batch_size = batchSize, epochs = nbEpochs, validation_data = (X_test, y_test))
    model.save(filepathToSave)

else:
    
    model = predictor.loadModel(filepath = 'convTest.h5')
    
    while True:
        print('\n')
        print('Use small letters only and use german symbols')
        word = input('Input your word: ')
        type(word)
        XTestSample = np.zeros((1,inputSize))
        for l in range(len(word)):
            letter = word[l]
            try:
                inx = alphabet.index(letter)
            except:
                sys.exit('This character is not a part of alphabet: ' + letter)
            XTestSample[0][l*len(alphabet) + inx] = 1
        XTestSample = XTestSample.reshape((XTestSample.shape[0],XTestSample.shape[1],1))
        prediction = model.predict(XTestSample)
        print('Chances of der: {:.2f}%'.format(prediction[0][0]*100))
        print('Chances of die: {:.2f}%'.format(prediction[0][1]*100))
        print('Chances of das: {:.2f}%'.format(prediction[0][2]*100))
    

        
    



