import numpy as np
import rvo2
from crowd_nav.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionXY


class RL(Policy):
    def __init__(self):

        super().__init__()
        self.name = 'RL'
        self.kinematics = 'holonomic'
        self.time_step = 0.25
        self.radius = 0.3
        self.max_speed = 1
        self.sim = None

    def configure(self, config):
        pass

    def set_phase(self, phase):
        pass

    def predict(self, state):
        pass
