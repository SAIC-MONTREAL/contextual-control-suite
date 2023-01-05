from dm_control.rl import control
from dm_control.utils import containers
from dm_control.utils import rewards
from dm_control.suite.reacher import Reacher, Physics, get_model_and_assets, _DEFAULT_TIME_LIMIT, _BIG_TARGET, _SMALL_TARGET
import contextual_control_suite.utils.rewards as utils

SUITE = containers.TaggedTasks()

@SUITE.add('benchmarking', 'easy')
def easy(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None, reward_kwargs=None):
  """Returns reacher with sparse reward with 5e-2 tol and randomized target."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = ReacherReward(target_size=_BIG_TARGET, random=random, reward_kwargs=reward_kwargs)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)

@SUITE.add('benchmarking')
def hard(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None, reward_kwargs=None):
  """Returns reacher with sparse reward with 1e-2 tol and randomized target."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = ReacherReward(target_size=_SMALL_TARGET, random=random,reward_kwargs=reward_kwargs)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)


class ReacherReward(Reacher):
    """A Reacher `Task` to balance the pole.
    Contains reward parameters compared to the original DeepMind Control task.
    """

    def __init__(self, target_size, random=None, reward_kwargs=None):
        """Initialize an instance of `Reacher`.
        Args:
        random: Optional, either a `numpy.random.RandomState` instance, an
            integer seed for creating a new `RandomState`, or None to select a seed
            automatically (default).
        """
        super().__init__(target_size, random=random)

        default_reward_parameters = {
            'finger_to_target': {
                'sigmoid': 'gaussian',
                'margin' : 0.5,
                'value_at_margin' : rewards._DEFAULT_VALUE_AT_MARGIN
            }
        }

        # update reward parameters
        self.reward_parameters = utils.set_reward_parameters(default_reward_parameters, reward_kwargs)

    def get_reward(self, physics):
        return rewards.tolerance(physics.finger_to_target_dist(), **self.reward_parameters['finger_to_target'])
