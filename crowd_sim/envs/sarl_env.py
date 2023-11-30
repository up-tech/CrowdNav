""""
Code base on CrowdNav: url
As basic class for crowd nav simulation 
"""
import gym
import math
import torch
from gym import spaces
import matplotlib.lines as mlines
import numpy as np
from matplotlib import patches
from numpy.linalg import norm
from crowd_sim.envs.utils.info import *
from collections import namedtuple
from crowd_sim.envs.utils.action import ActionXY, ActionRot
from crowd_sim.envs.utils.state import JointState
from crowd_sim.envs.basic_env import BasicEnv

class SARLEnv(BasicEnv):

    def __init__(self):
        super(SARLEnv, self).__init__()

        self.action_space = spaces.Discrete(81)
        self.observation_space = spaces.Box(low=-10, high=10, shape=(10, 13,))

        self.mapping_linear_x = lambda action: ((action % 9) - 4) / 4.0
        self.mapping_linear_y = lambda action: (math.floor(action / 9) - 4) / 4.0

        # space_dict = {}
        # for i in range(5):
        #     space_dict.update({f'human_{i}_ob' : spaces.Box(low=-5, high=5, shape=(13,))})

        # self.observation_space = spaces.Dict(space_dict)


    def configure(self, config):
        super().configure(config)

    def reset(self, seed = None):
        ob = super().reset(seed=seed)
        ob = self.transform_ob(ob)
        
        return ob
    
    def step(self, action):

        action_vx = self.mapping_linear_x(action)
        action_vy = self.mapping_linear_y(action)
        action_tuple = ActionXY(action_vx, action_vy)

        ob, reward, done, info = super().step(action_tuple)
        ob = self.transform_ob(ob)

        return ob, reward, done, info
    
    def robot_interface(self):
        return super().robot_interface()
 
    def render(self, mode=None):
        super().render(mode)

    def transform_ob(self, ob):
        state_tensor = torch.cat([torch.Tensor([ob[0] + human_state])
                            for human_state in ob[1:]], dim=0)
        for each_tensor in state_tensor:
            each_tensor = each_tensor[:14]

        local_ob = self.rotate(state_tensor)
        #print(local_ob)

        return local_ob
    
    def rotate(self, state):
        """
        Transform the coordinate to agent-centric.
        """

        # 'px', 'py', 'vx', 'vy', 'radius', 'gx', 'gy', 'v_pref', 'theta', 'px1', 'py1', 'vx1', 'vy1', 'radius1'
        #  0     1      2     3      4        5     6      7         8       9     10      11     12       13
        batch = state.shape[0]
        dx = (state[:, 5] - state[:, 0]).reshape((batch, -1))
        dy = (state[:, 6] - state[:, 1]).reshape((batch, -1))
        rot = torch.atan2(state[:, 6] - state[:, 1], state[:, 5] - state[:, 0])

        dg = torch.norm(torch.cat([dx, dy], dim=1), 2, dim=1, keepdim=True)
        v_pref = state[:, 7].reshape((batch, -1))
        vx = (state[:, 2] * torch.cos(rot) + state[:, 3] * torch.sin(rot)).reshape((batch, -1))
        vy = (state[:, 3] * torch.cos(rot) - state[:, 2] * torch.sin(rot)).reshape((batch, -1))

        radius = state[:, 4].reshape((batch, -1))

        # set theta to be zero since it's not used
        theta = torch.zeros_like(v_pref)
        vx1 = (state[:, 11] * torch.cos(rot) + state[:, 12] * torch.sin(rot)).reshape((batch, -1))
        vy1 = (state[:, 12] * torch.cos(rot) - state[:, 11] * torch.sin(rot)).reshape((batch, -1))
        px1 = (state[:, 9] - state[:, 0]) * torch.cos(rot) + (state[:, 10] - state[:, 1]) * torch.sin(rot)
        px1 = px1.reshape((batch, -1))
        py1 = (state[:, 10] - state[:, 1]) * torch.cos(rot) - (state[:, 9] - state[:, 0]) * torch.sin(rot)
        py1 = py1.reshape((batch, -1))
        radius1 = state[:, 13].reshape((batch, -1))
        radius_sum = radius + radius1
        da = torch.norm(torch.cat([(state[:, 0] - state[:, 9]).reshape((batch, -1)), (state[:, 1] - state[:, 10]).
                                reshape((batch, -1))], dim=1), 2, dim=1, keepdim=True)
        new_state = torch.cat([dg, v_pref, theta, radius, vx, vy, px1, py1, vx1, vy1, radius1, da, radius_sum], dim=1)

        return new_state