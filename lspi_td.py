import numpy as np
import math
from gym.spaces.discrete import Discrete
from PolynomialApproximator import PolynomialApproximator
from utils.AbsoluteSolver import AbsoluteSolver
from utils import plotting
from env import GameEnvironment
import os
from collections import namedtuple

class LSPITD():

    def __init__(self, env):

        """
            Creates solver and environment instances
        """

        self.env = env

        self.num_states = self.env.env.observation_space.shape[0]

        action_space = self.env.env.action_space

        if isinstance(action_space, Discrete):
            self.num_controls = 1
            self.num_control_values = action_space.n
            self.partitions = None
        else:
            delta = 0.1
            self.partitions = np.linspace(-1, 1, int(1 / delta))
            self.num_controls = action_space.shape[0]

        self.degree_of_polynomial = 2

        self.gamma = 0.9
        self.lambda_reg = 1

        # Create Solvers and approximators for each control

        self.solvers = []
        self.approximators = []
        for i in range(0, self.num_control_values):
            curr_solver = AbsoluteSolver(self.lambda_reg, self.gamma)
            curr_approximator = PolynomialApproximator(self.lambda_reg, self.gamma, self.degree_of_polynomial, curr_solver)

            self.solvers.append(curr_solver)
            self.approximators.append(curr_approximator)


        EpisodeStats = namedtuple(
            "Stats", ["episode_lengths", "episode_rewards"])
        self.stats = EpisodeStats(
            episode_lengths=np.zeros(10000),
            episode_rewards=np.zeros(10000))

    def policy(self, state):

        """
            returns a greedy policy using the approximated values so far for a given state
        """

        state = np.array(state).reshape((-1, 1))

        ctrl = None
        min_val = math.inf

        for i in range(0, self.num_control_values):
            action = i
            val = self.approximators[i].output(state)

            if val < min_val:
                min_val = val
                ctrl = action

        return ctrl, min_val

    def learn(self):

        """
            The train loop
        """
        num_iter = 0
        while True:

            def policyFn(state):
                ctrl, _ = self.policy(state)
                return ctrl

            def target(vec, next_vec):

                state_vec = vec
                predicted_ctrl = policyFn(state_vec)
                target_val = self.approximators[predicted_ctrl].output(next_vec)

                return target_val

            num_iter += 1
            ctrls, state, rewards = self.env.run_episode(policyFn)
            cost = -1 * rewards

            self.stats.episode_rewards[num_iter] = np.sum(rewards)
            self.stats.episode_lengths[num_iter] = rewards.shape[0]

            final_cost = rewards[-1]
            print("Episode - ", num_iter, " cost is - ", final_cost)

            has_converged = True
            for i in range(0, self.num_control_values):

                indices = np.where(ctrls == i)[0]

                filtered_state = state[indices, :]
                filtered_cost = cost[indices, :]

                self.approximators[i].update_features(filtered_state, filtered_cost, target)

                has_converged = has_converged and self.approximators[i].has_converged()

            if has_converged:
                print("Convergence check passed")
                break

            if num_iter == 1 or num_iter % 100 == 0:
                self.plot_q_fn("lspi_td" + str(num_iter))

            if (num_iter % 200) == 0 or num_iter == 1:

                path = "./models/lspi_td/all_" + str(self.degree_of_polynomial) + "/iter" + str(num_iter)
                self.save_progress(path)

                def policy(state):
                    action, _ = self.policy(state)
                    print("The action is - ", action)
                    return action

                gifs_path = "./gifs/lspi_td/all_2_" + str(self.degree_of_polynomial)
                if not os.path.exists(gifs_path):
                    os.makedirs(gifs_path)

                a, b, c = env.run_episode(policy, render=True, save_anim=True, save_path=gifs_path + "/iter_" + str(num_iter) + ".gif")

                # Save statistics
                dir_path = "./stats/lspi_td"
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                np.save(dir_path + "/episode_lengths",
                        self.stats.episode_lengths)
                np.save(dir_path + "/episode_rewards",
                        self.stats.episode_rewards)


        return True

    def plot_q_fn(self, identifier):

        """
            Generates plots of Q value
        """
        plotting.plot_cost(self.env.env, self.approximators, num_tiles=20, identifier=identifier)

    def save_progress(self, path):

        """
            Triggers the function approximator to save its internal state
        """

        if not os.path.exists(path):
            os.makedirs(path)

        for i in range(0, len(self.approximators)):
            self.approximators[i].save_state(path + "_" + str(i))

if __name__ == "__main__":

    GAME = 'LunarLander-v2'
    env = GameEnvironment(GAME)

    learning_obj = LSPITD(env)

    learning_obj.learn()
    learning_obj.plot_q_fn("after_training")


    def policy(state):
        action, _ = learning_obj.policy(state)
        print("The action is - ", action)
        return action


    a, b, c = env.run_episode(policy, render=True)

    print("This is the end")