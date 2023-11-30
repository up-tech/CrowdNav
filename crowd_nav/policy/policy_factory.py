from crowd_nav.policy.orca import ORCA
from crowd_nav.policy.sarl import SARL
from crowd_nav.policy.rl import RL
from gym import spaces

def none_policy():
    return None

policy_factory = dict()
policy_factory['orca'] = ORCA
policy_factory['sarl'] = SARL
policy_factory['rl'] = RL
policy_factory['none'] = none_policy