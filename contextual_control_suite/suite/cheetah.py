import copy
from dm_control.rl import control
from dm_control.utils import containers
from dm_control.utils import rewards
from dm_control.suite.cheetah import Cheetah, Physics, _DEFAULT_TIME_LIMIT, _RUN_SPEED
from dm_control.suite import common
from lxml import etree
import contextual_control_suite.utils.rewards as utils

SUITE = containers.TaggedTasks()


def get_model_and_assets(dynamics_kwargs=None):
  """Returns a tuple containing the model XML string and a dict of assets."""
  return _make_model(dynamics_kwargs), common.ASSETS


@SUITE.add('benchmarking')
def run(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None,
        reward_kwargs=None, dynamics_kwargs=None):
    """Returns the run task."""
    physics = Physics.from_xml_string(*get_model_and_assets(dynamics_kwargs))
    task = CheetahReward(random=random, reward_kwargs=reward_kwargs)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics, task, time_limit=time_limit,
                               **environment_kwargs)


def _make_model(dynamics_kwargs=None):
    """Generates an xml string defining with a modified torso."""
    xml_string = common.read_model('cheetah.xml')
    if dynamics_kwargs is None:
        return xml_string

    assert isinstance(dynamics_kwargs, dict)

    mjcf = etree.fromstring(xml_string)
    # Find the geom of the torso
    torso = mjcf.findall('./worldbody/body/geom')[0]
    head = mjcf.findall('./worldbody/body/geom')[1]
    bthigh = mjcf.findall('./worldbody/body/body')[0]
    fthigh = mjcf.findall('./worldbody/body/body')[1]
    if 'length' in dynamics_kwargs:
        torso.set('fromto', f"-{dynamics_kwargs['length']} 0 0 {dynamics_kwargs['length']} 0 0")
        head.set('pos', f"{dynamics_kwargs['length'] + 0.1} 0 0.1")
        bthigh.set('pos', f"-{dynamics_kwargs['length']} 0 0")
        fthigh.set('pos', f"{dynamics_kwargs['length']} 0 0")

    return etree.tostring(mjcf, pretty_print=True)


class CheetahReward(Cheetah):
    """A `Task` to train a running Cheetah."""

    def __init__(self, random=None, reward_kwargs=None):
        """Initialize an instance of `Cheetah`.
        Args:
        random: Optional, either a `numpy.random.RandomState` instance, an
            integer seed for creating a new `RandomState`, or None to select a seed
            automatically (default).
        """
        super().__init__(random=random)

        default_reward_parameters = {
            'speed': {
                'bounds': [_RUN_SPEED, float('inf')],
                'margin': _RUN_SPEED,
                'value_at_margin': rewards._DEFAULT_VALUE_AT_MARGIN,
                'sigmoid': 'linear'
            }
        }

        # update reward parameters
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
        """Returns a reward to the agent."""
        return rewards.tolerance(self.speed_direction * physics.speed(), **self.reward_parameters['speed'])
