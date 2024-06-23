#!/usr/bin/env python3
""" Q Learning """
import gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """ Load the pre-made FrozenLakeEnv evnironment from OpenAI’s gym """
    env = gym.make("FrozenLake-v0", is_slippery=is_slippery,
                   desc=desc, map_name=map_name)
    return env
