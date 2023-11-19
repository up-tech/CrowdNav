import os
import configparser
import numpy as np
import gym
import crowd_sim.envs.basic_env

def main():
    env_config = configparser.RawConfigParser()
    policy_config = configparser.RawConfigParser()
    env_config.read('crowd_nav/configs/sarl_env.config')
    policy_config.read('crowd_nav/configs/policy.config')

    env = gym.make('SARLEnv-v0')
    env.configure(env_config)

    total_episodes = 5
    init_seed = 0

    for ep in range(total_episodes):
        ep_seed = ep + init_seed
        ob = env.reset(seed=ep_seed)
        done = False
        while not done:
            action = env.robot_interface().act(ob)
            ob, reward, done, info = env.step(action)
        env.render()
    env.close()

if __name__ == '__main__':
    main()