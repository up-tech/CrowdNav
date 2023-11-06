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

    models_dir = "models/DQN"
    logdir = "logs"
    monitor_dir = "monitor"

    #models_path = f"{models_dir}/1000000.zip"
    models_path = monitor_dir + '/saved_model_model150000.zip'

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
    env.reset()

    model = DQN.load(models_path, env=env)

    episodes = 10
    for ep in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

        env.render(mode="video")

    env.close()

if __name__ == '__main__':
    main()