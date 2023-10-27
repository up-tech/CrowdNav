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
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common import results_plotter
import numpy as np

class SaveOnBestTrainingRewardCallback(BaseCallback):
    # save when total timesteps == check_freq * nproc
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), "timesteps")
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose >= 1:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose >= 1:
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path)

        return True

def main():

    models_dir = "models/DQN"
    logdir = "logs"
    monitor_log = "temp/"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    
    if not os.path.exists(monitor_log):
        os.makedirs(monitor_log, exist_ok=True)

    def make_env(env_id, rank, seed=0):
        def _init():
            env_config = configparser.RawConfigParser()
            env_config.read('crowd_nav/configs/env.config')

            robot = Robot(env_config, 'robot')

            env = gym.make(env_id)
            #set env config
            env.configure(env_config)
            env.set_robot(robot)
            env.reset(seed=seed+rank)

            env = Monitor(env, monitor_log)
            
            return env
        set_random_seed(seed=seed+rank)
        return _init

    policy_config = configparser.RawConfigParser()
    policy_config.read('crowd_nav/configs/policy.config')
    
    nproc = 20
    env = SubprocVecEnv([make_env('CrowdSim-v0', i) for i in range(nproc)])

    method = policy_config.get('policy', 'method')
    print(f'robot policy method: {method}')
    policy = policy_factory[method]

    policy_kwargs = dict(features_extractor_class=policy,
                         net_arch=[256, 256],
                         )

    params = {"learning_rate": 1e-3,
              "tensorboard_log": logdir,
              "batch_size": 100,
              "exploration_fraction": 0.5,
              "exploration_initial_eps": 0.6,
              "exploration_final_eps": 0.1,
              "target_update_interval": 5000,
              "learning_starts": 0,
             }

    model = DQN("MultiInputPolicy", env, verbose=1, **params, policy_kwargs=policy_kwargs)
    
    #callback = SaveOnBestTrainingRewardCallback(check_freq=1e+3, log_dir=callback_log)

    time_steps = int(2e+7)

    #model.learn(total_timesteps=time_steps, tb_log_name="DQN", callback=callback)
    model.learn(total_timesteps=time_steps, tb_log_name="DQN")

    model.save(f"{models_dir}/final_model")

    env.close()

if __name__ == '__main__':
    main()