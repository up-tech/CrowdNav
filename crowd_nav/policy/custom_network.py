import torch
import numpy as np
import torch.nn as nn
import torch.functional
from stable_baselines3 import DQN

class CustomNet(nn.Module):
    def __inti__(self, batch_size=32):
        super().__init__()
        self.fc1 = nn.Linear(in_features=13, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=1)
    def forward(self, state):
        fc1_output = self.fc1(state)
        fc1_output = nn.ReLU(fc1_output)
        fc2_output = self.fc2(fc1_output)
        fc2_output = nn.ReLU(fc2_output)
        output = self.fc3(fc2_output)
        return output
        