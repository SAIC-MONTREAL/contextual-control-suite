from dm_control.rl import control
from dm_control.utils import containers
from dm_control.utils import rewards
from dm_control.suite.pendulum import SwingUp, Physics, get_model_and_assets, _DEFAULT_TIME_LIMIT
import contextual_control_suite.utils.rewards as utils

SUITE = containers.TaggedTasks()


@SUITE.add('benchmarking')
def swingup(time_limit=_DEFAULT_TIME_LIMIT, random=None,
            environment_kwargs=None, reward_kwargs=None):
    """Returns pendulum swingup task ."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = SwingUpReward(random=random, reward_kwargs=reward_kwargs)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, **environment_kwargs)


class SwingUpReward(SwingUp):
    """A Pendulum `Task` to balance the pole.
    Contains reward parameters compared to the original DeepMind Control task.
    """

    def __init__(self, random=None, reward_kwargs=None):
        """Initialize an instance of `Pendulum`.
       Args:
         random: Optional, either a `numpy.random.RandomState` instance, an
           integer seed for creating a new `RandomState`, or None to select a seed
           automatically (default).
       """
        super().__init__(random=random)

        # default reward parameters in DM Control
        default_reward_parameters = {
            'upright': {
                'sigmoid': 'gaussian',
                'margin': 1,
                'value_at_margin': rewards._DEFAULT_VALUE_AT_MARGIN,
            },
            'small_velocity': {
                'sigmoid': 'gaussian',
                'margin': 5,
                'value_at_margin': rewards._DEFAULT_VALUE_AT_MARGIN
            }
        }

        # update reward parameters
        self.reward_parameters = utils.set_reward_parameters(default_reward_parameters, reward_kwargs)

    def get_reward(self, physics):
        """Non-sparse reward function for the pendulum task."""
        upright = (1 - physics.pole_vertical()) / 2
        upright = rewards.tolerance(upright,
                                    **self.reward_parameters['upright'])
        # upright = (1 + upright) / 2

        small_velocity = rewards.tolerance(physics.angular_velocity(),
                                           **self.reward_parameters['small_velocity']).min()
        small_velocity = (1 + small_velocity) / 2
        return upright * small_velocity
