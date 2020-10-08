from env import GameEnvironment
from lspi_td import LSPITD
import config
import numpy as np

GAME = 'LunarLander-v2'
env = GameEnvironment(GAME)
a,b,c = env.sample_run()

# # Implement LSPITD
