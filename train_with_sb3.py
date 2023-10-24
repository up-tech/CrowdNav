import os
import logging
import configparser
import torch
import gym
from stable_baselines3 import DQN
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import dummy_vec_env
from crowd_sim.envs.utils.robot import Robot
from crowd_nav.policy.policy_factory import policy_factory
from stable_baselines3.common.monitor import Monitor
from crowd_nav.policy.sarl_sb3 import CustomNN


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

    policy_kwargs = dict(features_extractor_class=CustomNN,
                         net_arch=[128, 50],
                         )

    params = {"learning_rate": 1e-3,
              "tensorboard_log": logdir,
              #"exploration_initial_eps": 0.5,
              "batch_size": 64,
              "learning_starts":200}

    #output dim == action discrete number
    #model = DQN("MultiInputPolicy", env, verbose=1, **params)
    model = DQN("MultiInputPolicy", env, verbose=1, **params, policy_kwargs=policy_kwargs)
    #model = DQN("MlpPolicy", env, verbose=2, **params, policy_kwargs=policy_kwargs) 
    print(model.policy)

    TIMESTEPS = 1000

    for ep in range(1, 10):
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="DQN")
        model.save(f"{models_dir}/{TIMESTEPS*ep}")

    env.close()

if __name__ == '__main__':
    main()