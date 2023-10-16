from crowd_sim.envs.policy.policy_factory import policy_factory
from crowd_nav.policy.cadrl import CADRL
from crowd_nav.policy.lstm_rl import LstmRL
from crowd_nav.policy.sarl import SARL
from crowd_nav.policy.modified_sarl import MODIFIED_SARL
from crowd_nav.policy.sarl_attn import SARL_ATTN
from crowd_nav.policy.sarl_self_attn import SARL_SELF_ATTN

policy_factory['cadrl'] = CADRL
policy_factory['lstm_rl'] = LstmRL
policy_factory['sarl'] = SARL
policy_factory['modified_sarl'] = MODIFIED_SARL
policy_factory['sarl_attn'] = SARL_ATTN
policy_factory['sarl_self_attn'] = SARL_SELF_ATTN