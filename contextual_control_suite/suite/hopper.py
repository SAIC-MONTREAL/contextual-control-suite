import copy
from dm_control.rl import control
from dm_control.utils import containers
from dm_control.utils import rewards
from dm_control.suite.hopper import Hopper, Physics, get_model_and_assets, _DEFAULT_TIME_LIMIT, _CONTROL_TIMESTEP, _STAND_HEIGHT, _HOP_SPEED
import contextual_control_suite.utils.rewards as utils

SUITE = containers.TaggedTasks()

@SUITE.add('benchmarking')
def stand(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None, reward_kwargs=None):
  """Returns a Hopper that strives to stand upright, balancing its pose."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = HopperReward(hopping=False, random=random, reward_kwargs=reward_kwargs)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)


@SUITE.add('benchmarking')
def hop(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None, reward_kwargs=None):
  """Returns a Hopper that strives to hop forward."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = HopperReward(hopping=True, random=random, reward_kwargs=reward_kwargs)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)


class HopperReward(Hopper):
    """A Hopper's `Task` to train a standing and a jumping Hopper."""

    def __init__(self, hopping, random=None, reward_kwargs=None):
        """Initialize an instance of `Hopper`.
        Args:
        hopping: Boolean, if True the task is to hop forwards, otherwise it is to
            balance upright.
        random: Optional, either a `numpy.random.RandomState` instance, an
            integer seed for creating a new `RandomState`, or None to select a seed
            automatically (default).
        """
        super().__init__(hopping, random=random)

        default_reward_parameters = {
            'height': {
                'bounds': [0.6, 2],
                'sigmoid': 'linear',
                'margin': 1,
                'value_at_margin': rewards._DEFAULT_VALUE_AT_MARGIN,
            },
            'speed': {
                'bounds': [2, float('inf')],
                'sigmoid': 'linear',
                'margin': 1,
                'value_at_margin': 0.5,
            },
            'control': {
                'sigmoid': 'quadratic',
                'margin': 1,
                'value_at_margin': rewards._DEFAULT_VALUE_AT_MARGIN,
            }
        }

        reward_kwargs_copy = copy.deepcopy(reward_kwargs)
        self.reward_parameters = utils.set_reward_parameters(default_reward_parameters, reward_kwargs_copy)

        # if margin is negative, change the speed direction
        if self.reward_parameters['speed']['margin'] < 0:
            self.reward_parameters['speed']['margin'] *= -1.0
            self.speed_direction = -1.0
        else:
            self.speed_direction = 1.0

        # manually overwrite the bounds
        self.reward_parameters['speed']['bounds'] = [self.reward_parameters['speed']['margin'], float('inf')]

        
    def get_reward(self, physics):
        """Returns a reward applicable to the performed task."""
        # standing = rewards.tolerance(physics.height(), **self.reward_parameters['height'])
        standing = rewards.tolerance(physics.height(), (_STAND_HEIGHT, 2))

        if self._hopping:
            hopping = rewards.tolerance(physics.speed(), **self.reward_parameters['speed'])
            return standing * hopping

        else:
            small_control = rewards.tolerance(physics.control(), **self.reward_parameters['control']).mean()
            small_control = (small_control + 4) / 5
            return standing * small_control
