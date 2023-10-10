import os
import logging
import configparser
import torch
import gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import dummy_vec_env
from crowd_sim.envs.utils.robot import Robot
from crowd_nav.policy.policy_factory import policy_factory

def main():
    env_config = configparser.RawConfigParser()
    policy_config = configparser.RawConfigParser()
    env_config.read('crowd_nav/configs/env.config')
    policy_config.read('crowd_nav/configs/policy.config')

    robot = Robot(env_config, 'robot')

    env = gym.make('CrowdSim-v0')
    #set env config
    env.configure(env_config)
    env.set_robot(robot)
    #env = dummy_vec_env([lambda: env])

    method = policy_config.get('policy', 'method')
    print(f'robot policy method: {method}')
    policy = policy_factory[method]()

    policy_kwargs = dict(
        net_arch=dict(pi=[], qf=policy)
    )

    #model = DQN("MlpPolicy", env, policy_kwargs=policy_kwargs, learning_rate=1e-3, verbose=1)
    model = DQN("MlpPolicy" , env, learning_rate=1e-3, verbose=1)
    print(model.policy)
    model.learn(total_timesteps=10000)

    env.close()

if __name__ == '__main__':
    main()