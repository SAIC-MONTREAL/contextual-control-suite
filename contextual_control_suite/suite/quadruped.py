import copy
from dm_control.rl import control
from dm_control.utils import containers
from dm_control.utils import rewards
from dm_control.suite.quadruped import Move, Escape, Physics, _upright_reward, make_model, _DEFAULT_TIME_LIMIT, \
    _CONTROL_TIMESTEP, _WALLS, _WALK_SPEED, _RUN_SPEED, _HEIGHTFIELD_ID
import contextual_control_suite.utils.rewards as utils
from lxml import etree
from dm_control.suite import common

SUITE = containers.TaggedTasks()


@SUITE.add()
def walk(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None, reward_kwargs=None):
    """Returns the Walk task."""
    xml_string = make_model(floor_size=_DEFAULT_TIME_LIMIT * _WALK_SPEED)
    physics = Physics.from_xml_string(xml_string, common.ASSETS)
    task = MoveReward(desired_speed=_WALK_SPEED, random=random, reward_kwargs=reward_kwargs)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics, task, time_limit=time_limit,
                               control_timestep=_CONTROL_TIMESTEP,
                               **environment_kwargs)


@SUITE.add()
def run(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None, reward_kwargs=None):
    """Returns the Run task."""
    xml_string = make_model(floor_size=_DEFAULT_TIME_LIMIT * _RUN_SPEED)
    physics = Physics.from_xml_string(xml_string, common.ASSETS)
    task = MoveReward(desired_speed=_RUN_SPEED, random=random, reward_kwargs=reward_kwargs)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics, task, time_limit=time_limit,
                               control_timestep=_CONTROL_TIMESTEP,
                               **environment_kwargs)


@SUITE.add()
def escape(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None, reward_kwargs=None):
    """Returns the Escape task."""
    xml_string = make_model(floor_size=40, terrain=True, rangefinders=True)
    physics = Physics.from_xml_string(xml_string, common.ASSETS)
    task = EscapeReward(random=random, reward_kwargs=reward_kwargs)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics, task, time_limit=time_limit,
                               control_timestep=_CONTROL_TIMESTEP,
                               **environment_kwargs)


class MoveReward(Move):
    def __init__(self, desired_speed, random=None, reward_kwargs=None):
        """Initializes an instance of `Move`.
        Args:
        desired_speed: A float. If this value is zero, reward is given simply
            for standing upright. Otherwise this specifies the horizontal velocity
            at which the velocity-dependent reward component is maximized.
        random: Optional, either a `numpy.random.RandomState` instance, an
            integer seed for creating a new `RandomState`, or None to select a seed
            automatically (default).
        """
        self.desired_speed = desired_speed

        super().__init__(desired_speed, random=random)

        default_reward_parameters = {
            'torso_velocity': {
                'sigmoid': 'linear',
                'margin': desired_speed,
                'value_at_margin': 0.5,
            }
        }
        reward_kwargs_copy = copy.deepcopy(reward_kwargs)
        self.reward_parameters = utils.set_reward_parameters(default_reward_parameters, reward_kwargs_copy)

        # if margin is negative, change the speed direction
        if self.reward_parameters['torso_velocity']['margin'] < 0:
            self.reward_parameters['torso_velocity']['margin'] *= -1.0
            self.speed_direction = -1.0
        else:
            self.speed_direction = 1.0

        # manually overwrite the bounds
        self.reward_parameters['torso_velocity']['bounds'] = [self.reward_parameters['torso_velocity']['margin'], float('inf')]

    def get_reward(self, physics):
        """Returns a reward to the agent."""

        # Move reward term.
        move_reward = rewards.tolerance(self.speed_direction * physics.torso_velocity()[0],
                                        **self.reward_parameters['torso_velocity'])

        return _upright_reward(physics) * move_reward


class EscapeReward(Escape):
    def __init__(self,random=None, reward_kwargs=None):

        super().__init__(random=random)

        default_reward_parameters = {
            'origin_distance': {
                'sigmoid': 'linear',
                'margin': 30,
                'value_at_margin': 0,
            }
        }

        self.reward_parameters = utils.set_reward_parameters(default_reward_parameters, reward_kwargs)
        self.reward_parameters['origin_distance']['bounds'] = [physics.model.hfield_size[_HEIGHTFIELD_ID, 0], float('inf')]

    def get_reward(self, physics):
        """Returns a reward to the agent."""

        # Escape reward term.
        escape_reward = rewards.tolerance(
            physics.origin_distance(), **self.reward_parameters['origin_distance'])

        return _upright_reward(physics, deviation_angle=20) * escape_reward
