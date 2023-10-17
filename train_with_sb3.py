import os
import logging
import configparser
import torch
import gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import dummy_vec_env
from crowd_sim.envs.utils.robot import Robot
from crowd_nav.policy.policy_factory import policy_factory
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn

def main():

    models_dir = "models/DQN"
    logdir = "logs"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    env_config = configparser.RawConfigParser()
    policy_config = configparser.RawConfigParser()
    env_config.read('crowd_nav/configs/env.config')
    policy_config.read('crowd_nav/configs/policy.config')

    robot = Robot(env_config, 'robot')

    env = gym.make('CrowdSim-v0')
    env = Monitor(env)
    #set env config
    env.configure(env_config)
    env.set_robot(robot)
    #env = dummy_vec_env([lambda: env])

    method = policy_config.get('policy', 'method')
    print(f'robot policy method: {method}')
    policy = policy_factory[method]()
    #print(model.policy)

    class CustomNN(BaseFeaturesExtractor):

        def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 128):
            super(CustomNN, self).__init__(observation_space, features_dim)

            self.nn = nn.Sequential(
                nn.Linear(13, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU()
            )

        def forward(self, observations: torch.Tensor) -> torch.Tensor:
            print(observations.shape)
            print('test')
            return self.nn(observations)

    policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                         features_extractor_class=CustomNN,
                         net_arch=[128, 128, 100])

    params = {"learning_rate": 1e-3,
              "tensorboard_log": logdir,
              "batch_size": 64}

    #model = DQN("MlpPolicy", env, policy_kwargs=policy_kwargs, learning_rate=1e-3, verbose=1)
    model = DQN("MultiInputPolicy", env, verbose=1, **params, policy_kwargs=policy_kwargs)
    print(model.policy)

    TIMESTEPS = 50000

    for ep in range(1, 10):
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="DQN")
        model.save(f"{models_dir}/{TIMESTEPS*ep}")

    env.close()

if __name__ == '__main__':
    main()