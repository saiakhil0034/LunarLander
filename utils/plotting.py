import matplotlib
import numpy as np
import pandas as pd
from collections import namedtuple
from matplotlib import pyplot as plt
import config
from matplotlib import animation
import os
from mpl_toolkits.mplot3d import Axes3D

EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards"])


def plot_cost_helper(env, estimator, other_defaults, num_tiles=20, identifier="", title="", index=""):
    """

    :param title: title of the plots
            e.g. horizontal speed = 1 with state components = 0
            other_defaults: values of other controls combined with x,y to make 8 dimensional state vector
                        - will be kept constant for all the (X, Y) datapoints
    :return: None
    """

    plt.close()
    x = np.linspace(max(env.observation_space.low[0], -5), min(
        env.observation_space.high[0], 5), num=num_tiles)
    y = np.linspace(max(env.observation_space.low[1], -5), min(
        env.observation_space.high[1], 5), num=num_tiles)
    X, Y = np.meshgrid(x, y)
    # Z = np.apply_along_axis(lambda _: np.max(estimator.output(np.concatenate((_, other_defaults)))), 2, np.dstack([X, Y]))

    if hasattr(estimator, '__call__'):
        func = estimator
    elif isinstance(estimator, list):
        def func(a):
            vals = [est.output(a) for est in estimator]
            return vals
    else:
        func = estimator.output

    Z_temp = np.apply_along_axis(lambda _: func(np.concatenate((_, other_defaults))), 2,
                                 np.dstack([X, Y]))

    # Plot the value function
    Z = np.min(Z_temp, axis=2).reshape((num_tiles, num_tiles))

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                           cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Value')
    ax.set_title(title)
    fig.colorbar(surf)

    folder = config.save_plots_base_path + "/" + identifier + "/" + index
    if not os.path.exists(folder):
        os.makedirs(folder)

    plt.savefig(folder + "/value_fn.png")
    # plt.show()

    # Plotting values for all other controls
    for control in range(0, Z_temp.shape[2]):

        Z = Z_temp[:, :, control].reshape((num_tiles, num_tiles))

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                               cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('Q Value')
        ax.set_title(title)
        fig.colorbar(surf)
        plt.savefig(folder + "/u=" + str(control) + ".png")
        # plt.show()

    print("Plot saved")


def plot_cost(env, estimator, num_tiles=20, identifier=""):
    """
    Wrapper around the plot_cost_helper function
    :param env:
    :param estimator: The predictor function that produces value estimate for each control
    :param num_tiles: density of sampling in the surface
    :param identifier: the base name to which other adjectives will be added
    :return:
    """

    titles = [
        "horizontal speed = 1 and others = 0",
        "vertical speed = 1 and others = 0",
        "angle = 1 and others = 0",
        "angular speed = 1 and others = 0",
        "first leg has contact = 1 and others = 0",
        "second leg has contact = 1 and others = 0"
    ]

    for i in range(0, 4):
        title = titles[i]
        other_defaults = np.zeros(6)
        other_defaults[i] = 1

        plot_cost_helper(env, estimator, other_defaults, num_tiles=num_tiles,
                         identifier=identifier, title=title, index=str(i))


def plot_episode_stats(stats, smoothing_window=10, noshow=False):

    """

    :param stats:
    :param smoothing_window:
    :param noshow:
    :return:
    """
    fig1 = plt.figure(figsize=(10,5))

    plt.plot(stats.episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    if noshow:
        plt.close(fig1)
    else:
        plt.show(fig1)

    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10, 5))
    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(
        smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(
        smoothing_window))
    if noshow:
        plt.close(fig2)
    else:
        plt.show(fig2)

    # Plot time steps and episode number
    fig3 = plt.figure(figsize=(10, 5))
    plt.plot(np.cumsum(stats.episode_lengths),
             np.arange(len(stats.episode_lengths)))
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title("Episode per time step")
    if noshow:
        plt.close(fig3)
    else:
        plt.show(fig3)

    return fig1, fig2, fig3

def save_frames_as_gif(frames, path='./animation.gif'):

    """
        Saves the given frames as gif
    :param frames:
    :param path:
    :return:
    """

    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(
        plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(path, writer='pillow', fps=60)
