import torch
import torch.nn as nn
from torch.nn.functional import softmax
import logging
from crowd_nav.policy.cadrl import mlp
from crowd_nav.policy.multi_human_rl import MultiHumanRL
import numpy as np

# def count_parameters(model):  # 传入的是模型实例对象
#     params = [p.numel() for p in model.parameters() if p.requires_grad]
#     for item in params:
#         print(f'{item:>16}')   # 参数大于16的展示
#     print(f'________\n{sum(params):>16}')  # 大于16的进行统计，可以自行修改

class ValueNetwork(nn.Module):
    def __init__(self, input_dim, self_state_dim, mlp1_dims, mlp2_dims, mlp3_dims, attention_dims, with_global_state,
                 cell_size, cell_num):
        super().__init__()
        self.self_state_dim = self_state_dim
        self.global_state_dim = mlp1_dims[-1]
        self.mlp1 = mlp(input_dim, mlp1_dims, last_relu=True)
        self.mlp2 = mlp(mlp1_dims[-1], mlp2_dims)
        self.with_global_state = with_global_state
        if with_global_state:
            self.attention = mlp(mlp1_dims[-1] * 2, attention_dims)
        else:
            self.attention = mlp(mlp1_dims[-1], attention_dims)
        self.cell_size = cell_size
        self.cell_num = cell_num
        mlp3_input_dim = mlp2_dims[-1] + self.self_state_dim
        self.mlp3 = mlp(mlp3_input_dim, mlp3_dims)
        self.attention_weights = None

        self.added_mlp = nn.Linear(100, 50)

        self.q_linear = nn.Linear(100, 50)
        self.k_linear = nn.Linear(100, 50)
        self.v_linear = nn.Linear(100, 50)

        self.lstm_input_dim = 7 # human feature size
        self.lstm_hidden_dim = 14 # hidden size
        self.lstm = nn.LSTM(self.lstm_input_dim, self.lstm_hidden_dim, batch_first=True)

    def forward(self, state):
        """
        First transform the world coordinates to self-centric coordinates and then do forward computation

        :param state: tensor of shape (batch_size, # of humans, length of a rotated state)
        :return:
        """
        #print(state[0, :, 8:9])
        
        #print(size[0])
        size = state.shape
        #print(size)

        robot_state = state[:, : 10, :self.self_state_dim]
        human_state = state[:, :, self.self_state_dim: ]
        human_state_seq = [None] * 10
        
        for i in range(10):
            human_state_seq[i] = human_state[:, i::10, :]

 #       count_parameters(self.lstm)

        h0 = torch.zeros(1, size[0], self.lstm_hidden_dim)
        c0 = torch.zeros(1, size[0], self.lstm_hidden_dim)

        hn_seq = [None] * 10
        for i in range(10):
            output, (hn_seq[i], cn) = self.lstm(human_state_seq[i], (h0, c0))
            hn_seq[i] = hn_seq[i].squeeze(0).unsqueeze(1)
        
        hn = torch.cat(hn_seq, dim=1)
        
        total_state = torch.cat([robot_state, hn], dim=-1)

        size = total_state.shape

        #print(f"total_state state size: {total_state.shape}")

        relu = nn.ReLU()
        
        self_state = state[:, 0, :self.self_state_dim]

        mlp1_output = self.mlp1(total_state.view((-1, size[2])))
        #print(mlp1_output.shape)
        #mlp2_output = self.mlp2(mlp1_output)

        if self.with_global_state:
            # compute attention scores
            global_state = torch.mean(mlp1_output.view(size[0], size[1], -1), 1, keepdim=True)
            global_state = global_state.expand((size[0], size[1], self.global_state_dim)).\
                contiguous().view(-1, self.global_state_dim)
            attention_input = torch.cat([mlp1_output, global_state], dim=1)
        else:
            attention_input = mlp1_output

        attention_input = attention_input.view(size[0], size[1], -1)
        #print(attention_input.shape)

        q = self.q_linear(attention_input)
        q = relu(q)
        k = self.k_linear(attention_input)
        k = relu(k)
        k = torch.transpose(k, -1, -2)

        scores = torch.bmm(q, k) #[100, 10, 10]
        temperature = 1 / np.sqrt(50)
        temperature = temperature / 0.05
        scores = torch.mul(scores, temperature) #[100, 10, 10]
        scores = torch.softmax(scores, dim=-1)
        scores = torch.sum(scores, dim=-2)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        self.attention_weights = weights[0, :, 0].data.cpu().numpy()
        
        v = self.v_linear(attention_input)
        v = relu(v)
        v = torch.transpose(v, -1, -2)

        e_h = torch.bmm(v, weights).squeeze(-1)
        
        # concatenate agent's state with global weighted humans' state
        joint_state = torch.cat([self_state, e_h], dim=1)
        value = self.mlp3(joint_state)
        return value


class SARL_LSTM_ATTN(MultiHumanRL):
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
        # self.model = ValueNetwork(self.input_dim(), self.self_state_dim, mlp1_dims, mlp2_dims, mlp3_dims,
        #                           attention_dims, with_global_state, self.cell_size, self.cell_num)
        self.model = ValueNetwork(56, self.self_state_dim, mlp1_dims, mlp2_dims, mlp3_dims,
                            attention_dims, with_global_state, self.cell_size, self.cell_num)
        self.multiagent_training = config.getboolean('sarl', 'multiagent_training')
        if self.with_om:
            self.name = 'OM-SARL'
        logging.info('Policy: {} {} global state'.format(self.name, 'w/' if with_global_state else 'w/o'))

    def get_attention_weights(self):
        return self.model.attention_weights
