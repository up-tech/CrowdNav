from gym.envs.registration import register

register(
    id='BasicEnv-v0',
    entry_point='crowd_sim.envs.basic_env:BasicEnv',
)

register(
    id='SARLEnv-v0',
    entry_point='crowd_sim.envs.sarl_env:SARLEnv',
)