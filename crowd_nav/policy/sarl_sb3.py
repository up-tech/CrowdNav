from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch
import gym

class CustomNN(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 128):
        super(CustomNN, self).__init__(observation_space, features_dim)
        
        self.linear_1 = nn.Linear(13, 128)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        
        obs_list = list(observations.values())
        input_tensor = torch.cat(obs_list) #(3, 13)

        output = self.linear_1(input_tensor)
        output = torch.sum(output, dim=0).unsqueeze(0)

        # output_dict = {}
        # output_dict["feature_extract"] = output
        #print(output.shape)
        
        return output
    
        #return self.nn(observations)