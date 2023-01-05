import copy
from dm_control.rl import control
from dm_control.utils import containers
from dm_control.utils import rewards
from dm_control.suite.walker import PlanarWalker, Physics, _DEFAULT_TIME_LIMIT, _CONTROL_TIMESTEP, _WALK_SPEED, _RUN_SPEED, _STAND_HEIGHT
from dm_control.suite import common
from lxml import etree
import contextual_control_suite.utils.rewards as utils

SUITE = containers.TaggedTasks()


def get_model_and_assets(dynamics_kwargs=None):
  """Returns a tuple containing the model XML string and a dict of assets."""
  return _make_model(dynamics_kwargs), common.ASSETS


@SUITE.add('benchmarking')
def stand(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None,
          reward_kwargs=None, dynamics_kwargs=None):
    """Returns the Stand task."""
    physics = Physics.from_xml_string(*get_model_and_assets(dynamics_kwargs))
    task = PlanarWalkerReward(move_speed=0, random=random, reward_kwargs=reward_kwargs)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs)


@SUITE.add('benchmarking')
def walk(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None,
         reward_kwargs=None, dynamics_kwargs=None):
    """Returns the Walk task."""
    physics = Physics.from_xml_string(*get_model_and_assets(dynamics_kwargs))
    task = PlanarWalkerReward(move_speed=_WALK_SPEED, random=random, reward_kwargs=reward_kwargs)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs)


@SUITE.add('benchmarking')
def run(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None,
        reward_kwargs=None, dynamics_kwargs=None):
    """Returns the Run task."""
    physics = Physics.from_xml_string(*get_model_and_assets(dynamics_kwargs))
    task = PlanarWalkerReward(move_speed=_RUN_SPEED, random=random, reward_kwargs=reward_kwargs)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs)


def _make_model(dynamics_kwargs=None):
    """Generates an xml string defining with a modified torso."""
    xml_string = common.read_model('walker.xml')
    if dynamics_kwargs is None:
        return xml_string

    assert isinstance(dynamics_kwargs, dict)

    mjcf = etree.fromstring(xml_string)
    # Find the geom of the torso
    torso = mjcf.find('./worldbody/body/geom')
    if 'length' in dynamics_kwargs:
        torso.set('size', f"0.07 {dynamics_kwargs['length']}")
    return etree.tostring(mjcf, pretty_print=True)


class PlanarWalkerReward(PlanarWalker):
    """A planar walker task."""

    def __init__(self, move_speed, random=None, reward_kwargs=None):
        """Initializes an instance of `PlanarWalker`.
        Args:
        move_speed: A float. If this value is zero, reward is given simply for
            standing up. Otherwise this specifies a target horizontal velocity for
            the walking task.
        random: Optional, either a `numpy.random.RandomState` instance, an
            integer seed for creating a new `RandomState`, or None to select a seed
            automatically (default).
        """
        super().__init__(move_speed, random=random)

        default_reward_parameters = {
            'horizontal_velocity': {
                'sigmoid': 'linear',
                'margin': move_speed / 2,
                'value_at_margin': 0.5
            },
        }
        reward_kwargs_copy = copy.deepcopy(reward_kwargs)
        self.reward_parameters = utils.set_reward_parameters(default_reward_parameters, reward_kwargs_copy)

        # if margin is negative, change the speed direction
        if self.reward_parameters['horizontal_velocity']['margin'] < 0:
            self.reward_parameters['horizontal_velocity']['margin'] *= -1.0
            self.speed_direction = -1.0
        else:
            self.speed_direction = 1.0

        # manually overwrite the bounds
        self.reward_parameters['horizontal_velocity']['bounds'] = [self.reward_parameters['horizontal_velocity']['margin'], float('inf')]

    def get_reward(self, physics):
        """Returns a reward to the agent."""

        standing = rewards.tolerance(physics.torso_height(),
                                     bounds=(_STAND_HEIGHT, float('inf')),
                                     margin=_STAND_HEIGHT / 2)

        upright = (1 + physics.torso_upright()) / 2
        stand_reward = (3*standing + upright) / 4

        if self._move_speed == 0:
            return stand_reward

        else:
            move_reward = rewards.tolerance(self.speed_direction * physics.horizontal_velocity(),
                                            **self.reward_parameters['horizontal_velocity'])
            return stand_reward * (5 * move_reward + 1) / 6
