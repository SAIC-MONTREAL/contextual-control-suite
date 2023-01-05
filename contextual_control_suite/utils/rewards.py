import collections.abc


def set_reward_parameters(default_reward_parameters, reward_kwargs):
    """If reward_kwargs has a key 'ALL', then all parameters are overwritten."""
    if reward_kwargs is None:
        return default_reward_parameters
    valid_reward_keys = set(default_reward_parameters.keys())
    valid_reward_keys.add('ALL')
    assert set(reward_kwargs.keys()) <= valid_reward_keys, "Invalid reward parameters."

    if 'ALL' in reward_kwargs.keys():
        reward_parameters = dict()
        for k in default_reward_parameters.keys():
            reward_parameters[k] = reward_kwargs['ALL']
    else:
        reward_parameters = update(default_reward_parameters, reward_kwargs)
    return reward_parameters


def update(d, u):
    """Updates nested dictionaries without overwriting missing items."""
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d
