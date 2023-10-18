from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch
import gym

class CustomNN(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super(CustomNN, self).__init__(observation_space, features_dim)

        # features_dim: output dim

        # self.nn = nn.Sequential(
        #     nn.Linear(13, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 128),
        #     nn.ReLU()
        # )

        self.linear_1 = nn.Linear(13, 128)
        
        #print('test')

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        
        output = self.linear_1(observations)
        print(observations.shape)
        return output
    
        #return self.nn(observations)