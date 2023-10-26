from crowd_nav.policy.orca import ORCA
from crowd_nav.policy.sarl import SARL


def none_policy():
    return None

policy_factory = dict()
policy_factory['orca'] = ORCA
policy_factory['sarl'] = SARL
policy_factory['none'] = none_policy