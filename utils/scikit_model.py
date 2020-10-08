#!/usr/bin/env python
# -*- coding: utf-8 -*-

########################################################################
# For project IV of ECE 276B,
# Authors       : A53279786,A53284020
# Last change   : June 11th,2020
# Professor     : Prof. Nikolay Atanasov
# Inspiration   : Q-value function approximation from https://github.com/dalmia/David-Silver-Reinforcement-learning/blob/master/Week%206%20-%20Value%20Function%20Approximations/Q-Learning%20with%20Value%20Function%20Approximation.py
########################################################################


import numpy as np
import sklearn
import sklearn.pipeline
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.kernel_approximation import RBFSampler
import os
from sklearn.externals import joblib


class ScikitModel(object):
    """ML model for finding u that minimises Q value """

    def __init__(self, env, name="scikit_rbf", load_from_disk=None):

        super(ScikitModel, self).__init__()

        # for Polynomial features
        self.feature_tr = PolynomialFeatures(degree=4)

        # For Rbf features
        # self.feature_tr = sklearn.pipeline.FeatureUnion([
        #     ("rbf1", RBFSampler(gamma=5.0, n_components=50)),
        #     ("rbf2", RBFSampler(gamma=2.0, n_components=50)),
        #     ("rbf3", RBFSampler(gamma=1.0, n_components=50)),
        #     ("rbf4", RBFSampler(gamma=0.5, n_components=50))
        # ])
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.env = env
        self.num_actions = self.env.action_space.n
        self.initialize_feature_transform()
        # print("hello")

        # Lets create a model for each action separately, so that it is easier to compare to find minimum later on
        self.models = []
        if not load_from_disk:
            for _ in range(env.action_space.n):
                model = SGDRegressor(learning_rate="optimal")
                model.partial_fit(self.get_features(env.reset()), [0])
                self.models.append(model)
        else:
            for i in range(env.action_space.n):
                model = joblib.load(f"{load_from_disk}/model_{i}.pkl")
                self.models.append(model)

        self.name = name

    def initialize_feature_transform(self):
        """To initialize the feature transform"""
        init_data = np.array([self.env.observation_space.sample()
                              for _ in range(5000)])

        self.scaler.fit(init_data)
        self.feature_tr.fit(self.scaler.transform(init_data))

    def get_features(self, state):
        """Method for getting the features for a state"""
        try:
            tr_state = self.scaler.transform([state])
        except:
            tr_state = self.scaler.transform(state)
        st_features = self.feature_tr.transform(tr_state)
        # print(st_features.shape)
        return st_features

    def predict(self, state):
        """Method for predicting Q-value for each state"""
        predictions = np.hstack([self.models[i].predict(
            self.get_features(state)).reshape(-1, 1) for i in range(self.env.action_space.n)])
        # print(predictions.shape)
        return predictions

    def fit(self, state, target):
        """for updating the model after collection of each data"""
        for action in range(self.num_actions):
            self.models[action].partial_fit(
                self.get_features(state), target[:, action])

    def save(self, num_it):
        """for saving the current mdoel"""
        print(os.getcwd())
        dir_path = f"./models/{self.name}/{num_it}"
        print(dir_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        for im, model in enumerate(self.models):
            file_path = dir_path + f"/model_{im}.pkl"
            joblib.dump(model, file_path)
