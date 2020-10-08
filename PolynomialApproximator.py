import numpy as np
from collections import namedtuple

class PolynomialApproximator():

    def __init__(self, lambda_reg, gamma, degree, solver):

        """
            lambda_reg : The regularization factor
            gamma: Discound factor
            degree: The degree of polynomial being fit
            solver: Solver instances
            parameters: column vector
        :param lr:
        """

        self.lambda_reg = lambda_reg
        self.gamma = gamma
        self.degree = degree
        self.solver = solver

        self.parameters = None
        self.convergence_Flag = False

    def initialize_parameters(self, number):
        self.parameters = np.random.normal(0, 0.5, number).reshape((-1, 1))
        self.eye = np.eye(number)

    def build_features(self, v):

        """
            Builds a polynomial of features - also includes cross terms
        :param v:
        :return:
        """
        features = v.reshape((-1, 1))
        features = np.concatenate((features, np.array([1]).reshape((-1, 1))), axis=0)

        for i in range(0, self.degree):
            res = features @ features.T
            indices = np.triu_indices(features.shape[0])
            features = res[indices[0], indices[1]].reshape((-1, 1))

        if self.parameters is None:
            self.initialize_parameters(features.shape[0])

        return features

    def output(self, data):

        features = self.build_features(data)
        output = features.T @ self.parameters

        return output[0]

    def has_converged(self):
        return self.convergence_Flag

    def update_features(self, data, cost, targetFn):

        """
            Updates the internal representation of features given an update
        """

        if self.solver.get_type() == "absolute":
            parameters = self.solver.step(data, cost, targetFn, self.build_features)

            if parameters is None:
                return True

            convergence_delta = np.max(np.abs(parameters - self.parameters))

            print("Convergence delta is - ", convergence_delta)

            if convergence_delta < 0.1:
                self.convergenceFlag = True

            self.parameters = parameters

        elif self.solver.get_type() == "partial":
            self.parameters += self.solver.step(data, cost, targetFn)

        return True

    def save_state(self, path):
        """
            Saves the current state of the environment
        """
        np.save(path, self.parameters)