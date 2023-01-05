from dm_control.rl import control
from dm_control.utils import containers
from dm_control.utils import rewards
from dm_control.suite import common
from lxml import etree
from dm_control.suite.cartpole import Balance, Physics, _DEFAULT_TIME_LIMIT
import contextual_control_suite.utils.rewards as utils

SUITE = containers.TaggedTasks()


def get_model_and_assets(dynamics_kwargs=None):
    """Returns a tuple containing the model XML string and a dict of assets."""
    return _make_model(dynamics_kwargs), common.ASSETS


@SUITE.add('benchmarking')
def balance(time_limit=_DEFAULT_TIME_LIMIT, random=None,
            environment_kwargs=None, reward_kwargs=None,
            dynamics_kwargs=None):
    """Returns the Cartpole Balance task."""
    physics = Physics.from_xml_string(*get_model_and_assets(dynamics_kwargs))
    task = BalanceReward(swing_up=False, random=random, reward_kwargs=reward_kwargs)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, **environment_kwargs)


@SUITE.add('benchmarking')
def swingup(time_limit=_DEFAULT_TIME_LIMIT, random=None,
            environment_kwargs=None, reward_kwargs=None,
            dynamics_kwargs=None):
    """Returns the Cartpole Swing-Up task."""
    physics = Physics.from_xml_string(*get_model_and_assets(dynamics_kwargs))
    task = BalanceReward(swing_up=True, random=random, reward_kwargs=reward_kwargs)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, **environment_kwargs)


def _make_model(dynamics_kwargs=None):
    """Generates an xml string defining with a modified mass."""
    xml_string = common.read_model('cartpole.xml')
    if dynamics_kwargs is None:
        return xml_string

    assert isinstance(dynamics_kwargs, dict)

    mjcf = etree.fromstring(xml_string)
    # Find the geom of the pole
    pole = mjcf.find('./worldbody/body/body/geom')
    if 'mass' in dynamics_kwargs:
        pole.set('mass', str(dynamics_kwargs['mass']))
    if 'size' in dynamics_kwargs:
        pole.set('size', str(dynamics_kwargs['size']))
    if 'length' in dynamics_kwargs:
        pole.set('fromto', f"0 0 0 0 0 {dynamics_kwargs['length']}")
    return etree.tostring(mjcf, pretty_print=True)


class BalanceReward(Balance):
    """A Cartpole `Task` to balance the pole.
    Contains reward parameters compared to the original DeepMind Control task.
    """

    def __init__(self, swing_up, random=None, reward_kwargs=None):
        """Initializes an instance of `Balance`.
        Args:
          swing_up: A `bool`, which if `True` sets the cart to the middle of the
            slider and the pole pointing towards the ground. Otherwise, sets the
            cart to a random position on the slider and the pole to a random
            near-vertical position.
          random: Optional, either a `numpy.random.RandomState` instance, an
            integer seed for creating a new `RandomState`, or None to select a seed
            automatically (default).
        """
        super().__init__(swing_up, sparse=False, random=random)
        assert isinstance(reward_kwargs, dict)

        # default reward parameters in DM Control
        default_reward_parameters = {
            'centered': {
                'sigmoid': 'gaussian',
                'margin': 2,
                'value_at_margin': rewards._DEFAULT_VALUE_AT_MARGIN,
            },
            'small_control': {
                'sigmoid': 'quadratic',
                'margin': 1,
                'value_at_margin': 0,
            },
            'small_velocity': {
                'sigmoid': 'gaussian',
                'margin': 5,
                'value_at_margin': rewards._DEFAULT_VALUE_AT_MARGIN
            }
        }

        # update reward parameters
        self.reward_parameters = utils.set_reward_parameters(default_reward_parameters, reward_kwargs)

    def _get_reward(self, physics, sparse):
        """"""
        upright = (physics.pole_angle_cosine() + 1) / 2

        centered = rewards.tolerance(physics.cart_position(),
                                     **self.reward_parameters['centered'])
        centered = (1 + centered) / 2
        small_control = rewards.tolerance(physics.control(),
                                          **self.reward_parameters['small_control'])[0]
        small_control = (4 + small_control) / 5
        small_velocity = rewards.tolerance(physics.angular_vel(),
                                           **self.reward_parameters['small_velocity']).min()
        small_velocity = (1 + small_velocity) / 2
        return upright.mean() * small_control * small_velocity * centered

    def get_reward(self, physics):
        """Returns a sparse or a smooth reward, as specified in the constructor."""
        return self._get_reward(physics, sparse=False)
