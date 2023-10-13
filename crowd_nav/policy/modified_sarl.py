import torch
import torch.nn as nn
from torch.nn.functional import softmax
import logging
from crowd_nav.policy.cadrl import mlp
from crowd_nav.policy.multi_human_rl import MultiHumanRL


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

        self.q_linear = nn.Linear(200, 128)
        self.k_linear = nn.Linear(200, 128)
        self.v_linear = nn.Linear(200, 128)

        self.score_linear = nn.Linear(128, 50)

        self.msa = torch.nn.MultiheadAttention(128, 8) #attention size: 128, num_heads: 8

    def forward(self, state):
        """
        First transform the world coordinates to self-centric coordinates and then do forward computation

        :param state: tensor of shape (batch_size, # of humans, length of a rotated state)
        :return:
        """
        size = state.shape
        #print(f'state size: {size}')
        self_state = state[:, 0, :self.self_state_dim]
        mlp1_output = self.mlp1(state.view((-1, size[2])))
        mlp2_output = self.mlp2(mlp1_output).view(size[0], size[1], -1)

        if self.with_global_state:
            # compute attention scores
            global_state = torch.mean(mlp1_output.view(size[0], size[1], -1), 1, keepdim=True)
            global_state = global_state.expand((size[0], size[1], self.global_state_dim)).\
                contiguous().view(-1, self.global_state_dim)
            #print(f'global_state size: {global_state.shape}')
            attention_input = torch.cat([mlp1_output, global_state], dim=1) #[batch_size*5, 200]
        else:
            attention_input = mlp1_output
        # modify start
        attention_input = attention_input.view(size[0], size[1], -1)
        #print(f'attention_input size: {attention_input.shape}')

        q = self.q_linear(attention_input)
        k = self.k_linear(attention_input)
        v = self.v_linear(attention_input)
        
        scores, _ = self.msa(q, k, v)
        # todo: should implement attention_weights
        #self.attention_weights = weights[0, :, 0].data.cpu().numpy()
        #print(f'scores size: {scores.shape}')
        scores = self.score_linear(scores)
    
        #weighted_humans = scores * mlp2_output
        scores = torch.sum(scores, dim=-1)
        scores = torch.nn.functional.softmax(scores, dim=-1).unsqueeze(-1)
        #print(f'scores size: {scores.shape}')
        self.attention_weights = scores[0, :, 0].data.cpu().numpy()
        #print(self.attention_weights)

        mlp2_output = torch.transpose(mlp2_output, -1, -2)
        #print(f'mlp2_output size: {mlp2_output.shape}')

        weighted_humans = torch.bmm(mlp2_output, scores)
        #print(f'weighted_humans size: {weighted_humans.shape}')

        #print(f'weighted_humans size: {weighted_humans.shape}')
        #weighted_humans = torch.sum(weighted_humans, dim=1)
        weighted_humans = torch.transpose(weighted_humans, -1, -2).squeeze(-2)
        #print(f'1weighted_humans size: {weighted_humans.shape}')

        joint_state = torch.cat([self_state, weighted_humans], dim=1)
        #print(f'joint_state size: {joint_state.shape}')

        #scores = self.attention(attention_input).view(size[0], size[1], 1).squeeze(dim=2)

        # masked softmax
        # weights = softmax(scores, dim=1).unsqueeze(2)
        #scores_exp = torch.exp(scores) * (scores != 0).float()
        #weights = (scores_exp / torch.sum(scores_exp, dim=1, keepdim=True)).unsqueeze(2)
        #self.attention_weights = weights[0, :, 0].data.cpu().numpy()

        # output feature is a linear combination of input features
        #features = mlp2_output.view(size[0], size[1], -1)
        # for converting to onnx
        # expanded_weights = torch.cat([torch.zeros(weights.size()).copy_(weights) for _ in range(50)], dim=2)
        #weighted_feature = torch.sum(torch.mul(weights, features), dim=1)

        #modify end

        # concatenate agent's state with global weighted humans' state
        #joint_state = torch.cat([self_state, weighted_feature], dim=1)
        value = self.mlp3(joint_state)
        return value


class MODIFIED_SARL(MultiHumanRL):
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
