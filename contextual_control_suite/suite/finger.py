import copy
from dm_control.rl import control
from dm_control.utils import containers
from dm_control.utils import rewards
from dm_control.suite.finger import Spin, Physics, _DEFAULT_TIME_LIMIT, _CONTROL_TIMESTEP, _SPIN_VELOCITY
from dm_control.suite import common
from lxml import etree
import contextual_control_suite.utils.rewards as utils

SUITE = containers.TaggedTasks()


def get_model_and_assets(dynamics_kwargs=None):
    """Returns a tuple containing the model XML string and a dict of assets."""
    return _make_model(dynamics_kwargs), common.ASSETS


@SUITE.add('benchmarking')
def spin(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None, reward_kwargs=None,
         dynamics_kwargs=None):
    """Returns the Spin task."""
    physics = Physics.from_xml_string(*get_model_and_assets(dynamics_kwargs))
    task = SpinReward(random=random, reward_kwargs=reward_kwargs)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs)


# needs to be tweaked
def _make_model(dynamics_kwargs=None):
    """Generates an xml string defining with a modified torso."""
    xml_string = common.read_model('finger.xml')
    if dynamics_kwargs is None:
        return xml_string

    assert isinstance(dynamics_kwargs, dict)

    mjcf = etree.fromstring(xml_string)
    # Find the geom of the torso
    distal = mjcf.findall('./worldbody/body/body/geom')[0]
    fingertip = mjcf.findall('./worldbody/body/body/geom')[1]

    spinner = mjcf.findall('./worldbody/body')[1]
    cap1 = spinner.findall('./geom')[0]
    cap2 = spinner.findall('./geom')[1]

    if 'length' in dynamics_kwargs:
        distal.set('fromto', f" 0 0 0 0 0 -{dynamics_kwargs['length']}")
        fingertip.set('fromto', f" 0 0 -{dynamics_kwargs['length'] - 0.03} 0 0 -{dynamics_kwargs['length'] + 0.001}")
        spinner.set('pos', f"{dynamics_kwargs['length'] + 0.04} 0 0.4")
        cap1.set('size', f"0.04 {dynamics_kwargs['length'] - 0.07}")
        cap2.set('size', f"0.04 {dynamics_kwargs['length'] - 0.07}")

    return etree.tostring(mjcf, pretty_print=True)


class SpinReward(Spin):
    """A Finger `Task` to spin the stopped body."""

    def __init__(self, random=None, reward_kwargs=None):
        """Initializes a new `Spin` instance.
        Args:
         random: Optional, either a `numpy.random.RandomState` instance, an
            integer seed for creating a new `RandomState`, or None to select a seed
            automatically (default).
        """

        super().__init__(random=random)

        # do we need bounds, sigmoid, value_at_margin (should it be the same as cheetah?)
        default_reward_parameters = {
            'spin': {
                'bounds': [_SPIN_VELOCITY, float('inf')],
                'margin': _SPIN_VELOCITY,
                'value_at_margin': rewards._DEFAULT_VALUE_AT_MARGIN,
                'sigmoid': 'linear'
            }
        }

        # update reward parameters
        reward_kwargs_copy = copy.deepcopy(reward_kwargs)
        self.reward_parameters = utils.set_reward_parameters(default_reward_parameters, reward_kwargs_copy)

        # if margin is negative, change the spin direction
        if self.reward_parameters['spin']['margin'] < 0:
            self.reward_parameters['spin']['margin'] *= -1.0
            self.spin_direction = -1.0
        else:
            self.spin_direction = 1.0

        # manually overwrite the bounds
        self.reward_parameters['spin']['bounds'] = [self.reward_parameters['spin']['margin'], float('inf')]

    def get_reward(self, physics):
        """Returns a reward to the agent."""
        return rewards.tolerance(self.spin_direction * physics.hinge_velocity()[0], **self.reward_parameters['spin'])
