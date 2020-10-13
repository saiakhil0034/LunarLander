#!/usr/bin/env python
# -*- coding: utf-8 -*-

########################################################################
# Last change   : June 11th,2020
# Inspiration   : Q-value function approximation from https://github.com/dalmia/David-Silver-Reinforcement-learning/blob/master/Week%206%20-%20Value%20Function%20Approximations/Q-Learning%20with%20Value%20Function%20Approximation.py
########################################################################

import os
import gym
import time
import random
import numpy as np
from collections import namedtuple
from env import GameEnvironment
from collections import deque

# For both SARSA and Q-learning only one line need to be changed in update model method of SARSA
# Current submission has Q-learning code


from utils.scikit_model import ScikitModel
from utils.nn_model import NeuralNetwork
from utils import plotting


EP_DECAY = 0.1


def save_obj(obj, dir_path, file_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    np.save(f"{dir_path}/{file_path}", obj)


class SARSA(object):
    """Class for SARSA algorithm"""

    def __init__(self, load_from_disk):
        super(SARSA, self).__init__()
        self.gamma = 0.9   # disount factor
        self.td_coeff = 0.5  # td learning coeff
        self._epsilon = 0.4  # for e-greedy policy
        self.env = gym.make('LunarLander-v2')
        self.genv = GameEnvironment("LunarLander-v2")
        # ScikitModel(self.env, load_from_disk=load_from_disk)
        self.func_approx = NeuralNetwork()
        # ScikitModel(self.env, load_from_disk=load_from_disk)
        # ScikitModel(self.env, load_from_disk=load_from_disk)
        # NeuralNetwork()  # for q-function approximation

        self.num_actions = self.env.action_space.n
        self.num_states = len(self.env.observation_space.sample())
        self.batch_size = 128
        self.num_episodes = 10000
        self.name = "sarsa_nn_1"
        # Keeps track of useful statistics
        EpisodeStats = namedtuple(
            "Stats", ["episode_lengths", "episode_rewards"])
        self.stats = EpisodeStats(
            episode_lengths=np.zeros(10000),
            episode_rewards=np.zeros(10000))

        # for experience reply, structural idea from cs221 stanford course
        self.cache = deque(maxlen=50000)

    def get_epsilon(self, num_it=0):
        """Epsilon for e-greedy method with decay"""
        return max((self._epsilon) / (num_it * EP_DECAY + 1), 0.01)

    def get_q_val(self, x):
        """ mathod to get Q value for given state x and control u"""
        self.func_approx.predict(x)

    def get_policy(self, state, epsilon, prob):
        """mathod for getting probabilities for each action
         in action space using epsilon greedy method"""

        q_values = self.func_approx.predict(state)
        best_action = np.argmax(q_values)
        rem_prob = 1.0 - prob.sum()
        prob[best_action] += rem_prob
        return prob

    def get_action(self, state):
        """For getting action from a state"""
        state = np.array(state).reshape(-1,
                                        self.num_states)  # self.num_states)
        return np.argmax(self.func_approx.predict(state))

    def updateCache(self, state, action, reward, newState, done):
        """Method for experience relay, this structure is inspired from cs 221 class tutorial"""
        self.cache.append((state, action, reward, newState, done))

    def update_model(self, epsilon, u_prob):
        # for experience replay, structural idea is from cs221 stanford course tutorial

        batch = random.sample(self.cache, self.batch_size)
        states = np.array([sample[0] for sample in batch])
        actions = np.array([sample[1] for sample in batch])
        rewards = np.array([sample[2] for sample in batch])
        newStates = np.array([sample[3] for sample in batch])
        dones = np.array([sample[4] for sample in batch])

        new_action_arr = []
        for new_state in newStates:
            new_action_arr.append(np.random.choice(self.num_actions,
                                                   p=self.get_policy(np.array(new_state).reshape(-1,
                                                                                                 self.num_states), epsilon, u_prob)))

        # initialize variable
        states = np.squeeze(states)
        newStates = np.squeeze(newStates)
        X = states
        y = self.func_approx.predict(states)
        ind = np.array([i for i in range(len(states))])
        # for SARSA

        targets = rewards + self.gamma * \
            self.func_approx.predict(newStates)[ind, [new_action_arr]]

        # for q learning
        # targets = rewards + self.gamma * \
        #     (np.amax(self.func_approx.predict(newStates), axis=1)) * (1 - dones)
        y[[ind], [actions]] = (1 - self.td_coeff) * \
            y[[ind], [actions]] + (self.td_coeff) * targets

        # update weight
        self.func_approx.fit(X, y)

    def optimise_for_goal(self, num_episodes=10000):
        """method for implementing actual sarsa algorithm"""
        ie = 0
        t0 = time.time()
        while (ie < num_episodes):
            # print(f"number of episodes created till now :{ie}")
            state = np.array(self.env.reset()).reshape(1, -1)
            epsilon = self.get_epsilon(ie)
            u_prob = np.full(self.num_actions, epsilon /
                             self.num_actions, dtype=float)
            term = False
            t1 = time.time()

            num_states_visited = 0

            while True:
                # sampling the action from the our epsilon greedy policy
                action = np.random.choice(
                    self.num_actions, p=self.get_policy(state, epsilon, u_prob))

                # Perform the action -> Get the reward and observe the next state
                new_s, reward, term, _ = self.env.step(action)
                new_s = np.array(new_s).reshape(1, -1)

                self.updateCache(state, action, reward, new_s, term)

                # Waiting till memory size is larger than batch size
                if len(self.cache) > self.batch_size:
                    # print("hi")
                    self.update_model(epsilon, u_prob)

                # stats
                self.stats.episode_rewards[ie] += reward
                self.stats.episode_lengths[ie] += 1

                # update current state
                state = new_s
                num_states_visited + + 1
                if term or num_states_visited > 500:
                    break

            # to visualize after some iterations
            if (ie % 200) == 0:
                dir_path = f'./models/{self.name}'
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                self.func_approx.save(f'{dir_path}/weights_{ie}.h5')
                dir_path = f"./stats/{self.name}"
                save_obj(self.stats.episode_lengths,
                         dir_path, "episode_lengths")
                save_obj(self.stats.episode_rewards,
                         dir_path, "episode_rewards")

                dir_path = f"./gifs/{self.name}"

                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                self.genv.run_episode(self.get_action, render=True, save_anim=True,
                                      save_path=f"{dir_path}/{ie}.gif")
                # Save plots

                # def prediction(a):
                #     return -1 * self.func_approx.predict(a)

                # plotting.plot_cost(self.env, prediction,
                #                    identifier="sarsa_eps_09")

            ie += 1
            t2 = time.time()
            print(
                f"episode {ie} took {t2-t1} seconds, total time : {t2-t0} seconds")


if __name__ == "__main__":
    algo = SARSA(load_from_disk=None)
    algo.optimise_for_goal()
