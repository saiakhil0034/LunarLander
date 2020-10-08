import numpy as np

class AbsoluteSolver():

    def __init__(self, lambda_reg, gamma):

        """
            lambda_reg : The regularization factor
        :param lr:
        """
        self.lambda_reg = lambda_reg
        self.gamma = gamma

    def get_type(self):

        """
            :return: the type of solver so that the updates can be handled appropriately
        """

        return "absolute"

    def step(self, data, cost, targetFn, feature_builder):

        """
            Runs on data and returns update
        :param: data
        :return: update
        """
        num_steps_in_episode = cost.shape[0]

        if num_steps_in_episode == 0 or num_steps_in_episode == 1:
            return None

        summation_vec = 0
        mat = 0
        for i in range(0, num_steps_in_episode - 1):
            features = feature_builder(data[i, :])

            summation_vec = summation_vec + features * cost[i]
            features_next_step = targetFn(data[i], data[i + 1])

            mat = mat + features @ (features - self.gamma * features_next_step).T

        params = np.linalg.inv(mat + self.lambda_reg * np.eye(mat.shape[0])) @ summation_vec

        params.reshape((-1, 1))

        return params