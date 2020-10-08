import gym
import numpy as np
from utils import plotting
import os

class GameEnvironment():

    """
        Wrapper around the open ai gym environment
    """

    def __init__(self, game):

        self.env_name = game
        self.env = gym.make(self.env_name)
        self.env.reset()

        """ For Lunar Lander continuous
        s (list): The state. Attributes:
                  s[0] is the horizontal coordinate
                  s[1] is the vertical coordinate
                  s[2] is the horizontal speed
                  s[3] is the vertical speed
                  s[4] is the angle
                  s[5] is the angular speed
                  s[6] 1 if first leg has contact, else 0
                  s[7] 1 if second leg has contact, else 0
        """

        self.episode_state = None
        self.episode_control = None
        self.episode_reward = None

    def reset(self):
        """
            resets the environment
        :return:
        """
        self.env.reset()
        self.reset_episode_state()

    def render(self, render=False, generate_frame=False):

        """
            renders a frame
        """

        frame = None

        if generate_frame:
            frame = self.env.render(mode="rgb_array")
        elif render:
            frame = self.env.render()

        return frame

    def get_state(self):
        """
        :return: the current state vector of environment
        """
        internal_state = self.env.lander

        pos = internal_state.position
        vel = internal_state.linearVelocity

        # VIEWPORT_W = self.env.viewer.width
        # VIEWPORT_H = self.env.viewer.height

        VIEWPORT_W = 600
        VIEWPORT_H = 400
        SCALE = 30.0  # affects how fast-paced the game is, forces should be adjusted as well
        LEG_DOWN = 18

        FPS = self.env.metadata['video.frames_per_second']

        state = [
            (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
            (pos.y - (self.env.helipad_y + LEG_DOWN / SCALE)) /
            (VIEWPORT_H / SCALE / 2),
            vel.x * (VIEWPORT_W / SCALE / 2) / FPS,
            vel.y * (VIEWPORT_H / SCALE / 2) / FPS,
            internal_state.angle,
            20.0 * internal_state.angularVelocity / FPS,
            1.0 if self.env.legs[0].ground_contact else 0.0,
            1.0 if self.env.legs[1].ground_contact else 0.0
        ]
        assert len(state) == 8

        return state

    def build_episode_state(self, control, state, reward):

        """
            Saves output of each step taken in the environment
        """
        if control is not None:
            if self.episode_control is not None:
                self.episode_control = np.concatenate(
                    (self.episode_control, np.array(control).reshape((1, -1))), axis=0)
            else:
                self.episode_control = np.array(control).reshape((1, -1))

        if state is not None:
            if self.episode_state is not None:
                self.episode_state = np.concatenate(
                    (self.episode_state, np.array(state).reshape((1, -1))), axis=0)
            else:
                self.episode_state = np.array(state).reshape((1, -1))

        if reward is not None:
            if self.episode_reward is not None:
                self.episode_reward = np.concatenate(
                    (self.episode_reward, np.array(reward).reshape((1, -1))), axis=0)
            else:
                self.episode_reward = np.array(reward).reshape((1, -1))

    def reset_episode_state(self):

        """
            clears environment state
        """

        self.episode_state = None
        self.episode_control = None
        self.episode_reward = None

    def run_episode(self, policyFn, render=False, save_anim=False, save_path=""):

        """
            Runs the episode using a given policy function
        """

        self.reset()

        episode_length = 0
        self.render(render)
        state = self.get_state()
        self.build_episode_state(None, state, None)

        frames = []

        while True:
            episode_length += 1
            frame = self.render(render, generate_frame=save_anim)

            if save_anim:
                frames.append(frame)

            action = policyFn(state)
            state, reward, done, info = self.env.step(action)

            self.build_episode_state(action, state, reward)

            # To check get_state computation correctness
            # match = self.get_state() == state
            # delta = np.array(self.get_state() - state)
            # delta = np.sum(np.abs(delta))
            # print("match result is ", match, "delta is ", delta)

            # print(state, reward, done, info)

            if done:
                print("End state is reached", episode_length)
                break

        self.env.close()

        if save_anim:
            try:
                plotting.save_frames_as_gif(frames, path=save_path)
            except:
                print("generating gif failed")

        return self.episode_control, self.episode_state, self.episode_reward

    def sample_run(self):

        """
            Runs an episode with uniformly sampled actions
        """

        def policy(state):
            return self.env.action_space.sample()  # take a random action

        self.run_episode(policy, render=True)

        return self.episode_control, self.episode_state, self.episode_reward

    # def evaluate_policy(self, policyFn):
        # run_episode(self, policyFn, render=False, save_anim=False, save_path=""):


if __name__ == "__main__":
    GAME = 'LunarLander-v2'
    env = GameEnvironment(GAME)
    a, b, c = env.sample_run()

    print("This is the end")