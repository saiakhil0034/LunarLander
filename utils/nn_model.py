#!/usr/bin/env python
# -*- coding: utf-8 -*-

import keras
from keras.models import Sequential
from keras.layers import Dense


class NeuralNetwork():
    """Class for neural network function approximator"""

    def __init__(self, batchSize=32, weights=None):
        self.model = Sequential()
        self.model.add(Dense(100, input_dim=8, activation='relu'))
        self.model.add(Dense(100, activation='relu'))
        self.model.add(Dense(4, activation='linear'))
        adam = keras.optimizers.adam(lr=0.001)
        self.model.compile(loss='mse', optimizer=adam)
        if isinstance(weights, str):
            self.model.load_weights(weights)

    def predict(self, state):
        return self.model.predict_on_batch(state)

    def fit(self, X, y):
        self.model.fit(X, y, epochs=1,  verbose=0)

    def save(self, weights):
        self.model.save_weights(weights)
