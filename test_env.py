import os
import configparser
import numpy as np
import gym
import crowd_sim
from crowd_sim.envs.utils.robot import Robot
from crowd_nav.policy.policy_factory import policy_factory

def main():
    env_config = configparser.RawConfigParser()
    policy_config = configparser.RawConfigParser()
    env_config.read('crowd_nav/configs/env.config')
    policy_config.read('crowd_nav/configs/policy.config')

    env = gym.make('CrowdSim-v0')
    #set env config
    env.configure(env_config)

    #set robot policy and load robot to env
    method = policy_config.get('policy', 'method')
    print(f'robot policy method: {method}')
    policy = policy_factory[method]()
    robot = Robot(env_config, 'robot')
    robot.set_policy(policy)
    env.set_robot(robot)
    #policy.set_env(env)

    ob = env.reset()
    print(ob)
    done = False

    while not done:
        action = robot.act(ob)
        ob, _, done, info = env.step(action)

    env.render('video', None)
    env.close()

if __name__ == '__main__':
    main()