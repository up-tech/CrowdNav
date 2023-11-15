import os
import configparser
import numpy as np
import gym
import crowd_sim.envs.basic_env

def main():
    env_config = configparser.RawConfigParser()
    policy_config = configparser.RawConfigParser()
    env_config.read('crowd_nav/configs/env.config')
    policy_config.read('crowd_nav/configs/policy.config')

    env = gym.make('BasicEnv-v0')
    env.configure(env_config)

    total_episodes = 5
    init_seed = 0

    for ep in range(total_episodes):
        ep_seed = ep + init_seed
        print(ep_seed)
        ob = env.reset(seed=ep_seed)
        done = False
        while not done:
            action = env.robot_interface().act(ob)
            ob, _, done, info = env.step(action)
        env.render('video')
    env.close()

if __name__ == '__main__':
    main()