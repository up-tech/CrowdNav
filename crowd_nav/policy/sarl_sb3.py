from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch
import gym

class CustomNN(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 128):
        super(CustomNN, self).__init__(observation_space, features_dim)

        self.mlp1 = nn.Sequential(
          nn.Linear(13, 150),
          nn.ReLU(),
          nn.Linear(150, 100),
          nn.ReLU()
        )
        self.mlp2 = nn.Sequential(
          nn.Linear(100, 100),
          nn.ReLU(),
          nn.Linear(100, 50),
        )        
        self.mlp3 = nn.Sequential(
          nn.Linear(56, 150),
          nn.ReLU(),
          nn.Linear(150, features_dim),
        )
        self.attention = nn.Sequential(
          nn.Linear(100, 100),
          nn.ReLU(),
          nn.Linear(100, 100),
          nn.ReLU(),
          nn.Linear(100, 1),
        )
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        #print(f"dict len: {len(observations)}")
        obs_list = []
        for key in observations.keys():
            obs_list.append(observations[key].unsqueeze(1))
        state = torch.cat(obs_list, dim=1)
        size = state.shape
        print(f"obs tensor shape: {size}")

        self_state = state[:, 0, :6]
        mlp1_output = self.mlp1(state.view((-1, size[2])))
        mlp2_output = self.mlp2(mlp1_output)
        attention_input = mlp1_output
        scores = self.attention(attention_input).view(size[0], size[1], 1).squeeze(dim=2)
        scores_exp = torch.exp(scores) * (scores != 0).float()
        weights = (scores_exp / torch.sum(scores_exp, dim=-1, keepdim=True)).unsqueeze(2)
        features = mlp2_output.view(size[0], size[1], -1)
        weighted_feature = torch.sum(torch.mul(weights, features), dim=1)
        print(f"weighted_feature shape: {weighted_feature.shape}")
        joint_state = torch.cat([self_state, weighted_feature], dim=1)
        value = self.mlp3(joint_state)

        #self.attention_weights = weights[0, :, 0].data.cpu().numpy()
    
        print(value.shape)
        
        return value