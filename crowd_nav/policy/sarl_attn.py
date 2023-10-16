import torch
import torch.nn as nn
from torch.nn.functional import softmax
import logging
from crowd_nav.policy.cadrl import mlp
from crowd_nav.policy.multi_human_rl import MultiHumanRL
import numpy as np


class ValueNetwork(nn.Module):
    def __init__(self, input_dim, self_state_dim, mlp1_dims, mlp2_dims, mlp3_dims, attention_dims, with_global_state,
                 cell_size, cell_num):
        super().__init__()
        self.self_state_dim = self_state_dim
        self.global_state_dim = mlp1_dims[-1]
        self.mlp1 = mlp(input_dim, mlp1_dims, last_relu=True)
        self.mlp2 = mlp(mlp1_dims[-1], mlp2_dims)
        self.cell_size = cell_size
        self.cell_num = cell_num
        mlp3_input_dim = mlp2_dims[-1] + self.self_state_dim
        self.mlp3 = mlp(mlp3_input_dim, mlp3_dims)

        #[batch_size, human_size, 13]

        self.phi_1 = nn.Linear(6, 128)
        self.phi_2 = nn.Linear(13, 128)
        self.phi_3 = nn.Linear(13, 128)

        self.q_linear = nn.Linear(128, 128)
        self.k_linear = nn.Linear(128, 128)
        self.v_linear = nn.Linear(128, 128)

        self.final_linear = nn.Linear(256, 1)

    def forward(self, state):
        
        relu = nn.ReLU()
        size = state.shape

        self_state = state[:, 0, :self.self_state_dim] #[100, 6]
        phi_1_output = self.phi_1(self_state) #[100, 128]
        e_h = phi_1_output
        
        phi_2_output = self.phi_2(state) #[100, 10, 13] -> [100, 10, 128]
        phi_2_output = relu(phi_2_output)
        attn_v = self.v_linear(phi_2_output)

        #print(phi_2_output[0][0])

        phi_3_output = self.phi_3(state)
        phi_3_output = relu(phi_3_output)
        #print(state[0])
        
        attn_q = self.q_linear(phi_3_output) #[100, 10, 128]
        attn_q = relu(attn_q)
        #print(attn_q[0][0])
        attn_k = self.k_linear(phi_3_output) #[100, 10, 128]
        attn_k = relu(attn_k)

        attn_k = torch.transpose(attn_k, -1, -2)
        scores = torch.bmm(attn_q, attn_k) #[100, 10, 10]
        #print(scores[0][0])
        temperature = 1 / np.sqrt(128)
        scores = torch.mul(scores, temperature) #[100, 10, 10]
        #print(scores[0][0])
        scores = torch.sum(scores, dim=-2) #[100, 10]
        scores = relu(scores)

        weights = torch.softmax(scores, dim=-1).unsqueeze(-1) #[100, 10, 1]
        #print(weights[0])
        #print(weights.shape)
        self.attention_weights = weights[0, :, 0].data.cpu().numpy()
        #print(self.attention_weights)

        attn_v = torch.transpose(attn_v, -1, -2) #[100, 10, 13] -> [100, 128, 10]

        e_g = torch.bmm(attn_v, weights).squeeze(-1) #[100, 128, 10] * [100, 10, 1]

        embedded_state = torch.cat([e_g, e_h], dim=-1) #[100, 256]

        value = self.final_linear(embedded_state) #[100, 1]
    
        return value


class SARL_ATTN(MultiHumanRL):
    def __init__(self):
        super().__init__()
        self.name = 'SARL'

    def configure(self, config):
        self.set_common_parameters(config)
        mlp1_dims = [int(x) for x in config.get('sarl', 'mlp1_dims').split(', ')]
        mlp2_dims = [int(x) for x in config.get('sarl', 'mlp2_dims').split(', ')]
        mlp3_dims = [int(x) for x in config.get('sarl', 'mlp3_dims').split(', ')]
        attention_dims = [int(x) for x in config.get('sarl', 'attention_dims').split(', ')]
        self.with_om = config.getboolean('sarl', 'with_om')
        with_global_state = config.getboolean('sarl', 'with_global_state')
        self.model = ValueNetwork(self.input_dim(), self.self_state_dim, mlp1_dims, mlp2_dims, mlp3_dims,
                                  attention_dims, with_global_state, self.cell_size, self.cell_num)
        self.multiagent_training = config.getboolean('sarl', 'multiagent_training')
        if self.with_om:
            self.name = 'OM-SARL'
        logging.info('Policy: {} {} global state'.format(self.name, 'w/' if with_global_state else 'w/o'))

    def get_attention_weights(self):
        return self.model.attention_weights
