import genesis as gs
import imageio
import pytest

from genesis_gymnasium.envs.chase.chase_env import ChaseEnv


@pytest.fixture
def chase_env():
    env = ChaseEnv()
    return env


def test_chase_env_init(chase_env):
    env = chase_env
    assert isinstance(env, ChaseEnv)
    assert isinstance(env.scene, gs.Scene)
    assert env.render_mode == "rgb_array"


def test_chase_env_render(chase_env):
    image_save_path = "tests/out/test_chase_env_render.png"
    env = chase_env
    rgb_array, _, _, _ = env.render()
    imageio.imsave(image_save_path, rgb_array)
