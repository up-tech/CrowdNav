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
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env

def make_env(env_id, rank, seed=0):
    def _init():
        env_config = configparser.RawConfigParser()
        env_config.read('crowd_nav/configs/env.config')

        robot = Robot(env_config, 'robot')

        env = gym.make(env_id)
        #set env config
        env.configure(env_config)
        env.set_robot(robot)

        env = Monitor(env)
        return env
    set_random_seed(seed=seed+rank)
    return _init

def main():

    models_dir = "models/DQN"
    logdir = "logs"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    
    policy_config = configparser.RawConfigParser()
    policy_config.read('crowd_nav/configs/policy.config')
    
    nproc = 20
    env = SubprocVecEnv([make_env('CrowdSim-v0', i) for i in range(nproc)])

    method = policy_config.get('policy', 'method')
    print(f'robot policy method: {method}')
    policy = policy_factory[method]
    print(model.policy)

    policy_kwargs = dict(features_extractor_class=policy,
                         net_arch=[128],
                         )

    params = {"learning_rate": 1e-3,
              "tensorboard_log": logdir,
              "batch_size": 64,
              #"exploration_initial_eps": 0.5,
              #"learning_starts":200,
             }

    model = DQN("MultiInputPolicy", env, verbose=1, **params, policy_kwargs=policy_kwargs)
    #model = DQN("MlpPolicy", env, verbose=2, **params, policy_kwargs=policy_kwargs) 
    #print(model.policy)

    time_steps = int(2e+5)

    for ep in range(1, 30):
        model.learn(total_timesteps=time_steps, reset_num_timesteps=False, tb_log_name="DQN")
        model.save(f"{models_dir}/{time_steps*ep}")

    env.close()

if __name__ == '__main__':
    main()