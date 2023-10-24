from crowd_nav.policy.linear import Linear
from crowd_nav.policy.orca import ORCA
#from crowd_nav.policy.cadrl import CADRL
#from crowd_nav.policy.lstm_rl import LstmRL
#from crowd_nav.policy.sarl import SARL
#from crowd_nav.policy.sarl import ValueNetwork


def none_policy():
    return None

policy_factory = dict()
policy_factory['linear'] = Linear
policy_factory['orca'] = ORCA
policy_factory['none'] = none_policy
#policy_factory['cadrl'] = CADRL
#policy_factory['lstm_rl'] = LstmRL
#policy_factory['sarl'] = ValueNetwork