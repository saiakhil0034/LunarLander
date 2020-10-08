import numpy as np
import sklearn.preprocessing.PolynomialFeatures


class MLModel(object):
    """ML model for finding u that minimises Q value """

    def __init__(self, args):
        super(MLModel, self).__init__()
        self.arg = arg

        self.poly = PolynomialFeatures(degree=args["degree"])
        num_features = self.poly.fit_transform(
            np.random.rand(5, 7)).shape[1] + 2
        self.weights = np.random.rand(num_features, 1)

    def predict(self, features):
        return features.dot(weights)

    def train(self, data, label):
        features = np.hstack([self.poly.fit_transform(data[:-2]), data[-2:]])
        self.weights = np.linalg.ing(features.dot(
            features.T)).dot((features.T).dot(label))
        print("weights updated")
