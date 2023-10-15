from stable_baselines3.common.env_checker import check_env
from crowd_sim.envs.crowd_sim import CrowdSim

env = CrowdSim()
check_env(env)