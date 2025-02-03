from abc import ABC
from typing import Any, TypeAlias, Literal

import genesis as gs
import gymnasium as gym
from gymnasium.core import ObsType, ActType
import numpy as np
from numpy.typing import NDArray
import torch

RenderMode: TypeAlias = Literal[
    "rgb_array",
    "depth_array",
    "seg_array",
    "colorized_seg_array",
    "normal_array",
    "all_arrays",
    "human",
]

RenderOutput: TypeAlias = tuple[
    NDArray[np.uint8] | None,
    NDArray[np.uint8] | None,
    NDArray[np.uint8] | None,
    NDArray[np.uint8] | None,
]

DEFAULT_SCENE: gs.Scene = gs.Scene(
    show_viewer=False,
    viewer_options=gs.options.ViewerOptions(
        res=(1280, 960),
        camera_pos=(3.5, 0.0, 2.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=40,
        max_FPS=60,
    ),
    vis_options=gs.options.VisOptions(
        show_world_frame=True,  # visualize the coordinate frame of `world` at its origin
        world_frame_size=1.0,  # length of the world frame in meter
        show_link_frame=False,  # do not visualize coordinate frames of entity links
        show_cameras=False,  # do not visualize mesh and frustum of the cameras added
        plane_reflection=True,  # turn on plane reflection
        ambient_light=(0.1, 0.1, 0.1),  # ambient light setting
    ),
    renderer=gs.renderers.Rasterizer(),  # using rasterizer for camera rendering
)


class GenesisEnv(gym.Env, ABC):
    """
    Superclass for all Genesis environments.
    """

    metadata: dict[str, Any] = {
        "render_modes": [
            "rgb_array",
            "depth_array",
            "seg_array",
            "colorized_seg_array",
            "normal_array",
            "all_arrays",
            "human",
        ],
    }

    def __init__(
        self, scene: gs.Scene = DEFAULT_SCENE, render_mode: str = "rgb_array"
    ) -> None:
        """
        Initializes the Genesis environment.

        Parameters
        ----------
        """
        self.scene: gs.Scene = scene
        self._add_entities()
        self._add_camera()
        self.scene.build()

        self.render_mode: str = render_mode
        if self.render_mode != "human":
            for cam in self.scene._visualizer._cameras:
                assert (
                    cam._GUI == False
                ), "GUI must be False for non-human render modes."

    def _add_entities(self) -> None:
        """
        Adds entities to the scene.
        """
        raise NotImplementedError

    def _add_camera(self) -> None:
        """
        Adds camera to the scene.
        """
        raise NotImplementedError

    def step(
        self, action: ActType
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        """
        Executes one time step within the environment.

        Parameters
        ----------
        action : gymnasium.core.ActType
            The action to take within the environment.

        Returns
        -------
        observation: gymnasium.core.ObsType
            The observation of the environment.
        reward: float
            The reward of the environment.
        terminated: bool
            Whether the episode has terminated.
        truncated: bool
            Whether the episode was truncated.
        info: dict[str, Any]
            Additional information.
        """

        self.scene.step()

    def render(
        self,
    ) -> RenderOutput | list[RenderOutput]:
        """
        Renders the environment.

        Returns
        -------
        output: RenderOutput | list[RenderOutput]
            The rendered output.
            Contains the RGB, depth, segmentation, and normal arrays.
            If there is only one camera, returns a single RenderOutput.
            If there are multiple cameras, returns a list of RenderOutputs.
        """

        rgb: bool = (
            True
            if self.render_mode == "rgb_array" or "human" or "all_arrays"
            else False
        )
        depth: bool = (
            True
            if self.render_mode == "depth_array" or "human" or "all_arrays"
            else False
        )
        segmentation: bool = (
            True
            if self.render_mode == "seg_array" or "human" or "all_arrays"
            else False
        )
        colorized_seg: bool = (
            True if self.render_mode == "colorized_seg_array" else False
        )
        normal: bool = (
            True
            if self.render_mode == "normal_array" or "human" or "all_arrays"
            else False
        )
        output: list[RenderOutput] = []
        for cam in self.scene._visualizer._cameras:
            output.append(cam.render(rgb, depth, segmentation, colorized_seg, normal))

        if len(output) == 0:
            raise ValueError("No cameras found in the scene.")
        elif len(output) == 1:
            return output[0]
        else:
            return output
