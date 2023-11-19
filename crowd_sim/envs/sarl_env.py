""""
Code base on CrowdNav: url
As basic class for crowd nav simulation 
"""
import gym
from gym import spaces
import matplotlib.lines as mlines
import numpy as np
from matplotlib import patches
from numpy.linalg import norm
from crowd_sim.envs.utils.info import *
from collections import namedtuple
from crowd_sim.envs.utils.action import ActionXY, ActionRot
from crowd_sim.envs.basic_env import BasicEnv

class SARLEnv(BasicEnv):

    def __init__(self):
        super(SARLEnv, self).__init__()

    def configure(self, config):
        super().configure(config)

    def reset(self, seed = None):
        ob = super().reset(seed=seed)

        # transform_ob
        return ob
    
    def step(self, action):

        ob, reward, done, info = super().step(action)
        # transform_ob

        return ob, reward, done, info
    
    def robot_interface(self):
        return super().robot_interface()
 
    def render(self, mode=None):
        super().render(mode)

    