import argparse
import numpy as np

from contextual_control_suite import suite
from dm_control import viewer


parser = argparse.ArgumentParser()
parser.add_argument('--domain', type=str, default='cheetah')
parser.add_argument('--task', type=str, default='run')
parser.add_argument('--steps', type=int, default=1000)
args = parser.parse_args()

# Reward parameters
reward_kwargs = {
    'ALL': {
        'sigmoid': 'linear',
        'margin': 10,
    },
}

# Dynamics parameters
dynamics_kwargs = {
    'length': 0.5
}

task_kwargs = {
    'reward_kwargs': reward_kwargs,
    'dynamics_kwargs': dynamics_kwargs
}

# Create the environment with the custom task parameters
env = suite.load(args.domain, args.task, task_kwargs=task_kwargs)
action_spec = env.action_spec()


def random_policy(time_step):
    del time_step
    return np.random.uniform(low=action_spec.minimum,
                             high=action_spec.maximum,
                             size=action_spec.shape)


# Run the environment without the viewer
# time_step = env.reset()
# for t in range(args.steps):
#     if time_step.last():
#         time_step = env.reset()
#     action = random_policy(time_step)
#     time_step = env.step(action)
#     print(f"Reward: {time_step.reward:.2f}")

# Run the environment with the viewer
viewer.launch(env, policy=random_policy)
