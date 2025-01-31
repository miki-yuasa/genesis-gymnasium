from abc import ABC
from typing import Any, TypeAlias

import genesis as gs
import gymnasium as gym
from gymnasium.core import ObsType, ActType
import numpy as np
from numpy.typing import NDArray
import torch

RenderOutput: TypeAlias = tuple[
    NDArray[np.uint8] | None,
    NDArray[np.uint8] | None,
    NDArray[np.uint8] | None,
    NDArray[np.uint8] | None,
]


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

    def __init__(self, scene: gs.Scene, render_mode: str = "rgb_array") -> None:
        """
        Initializes the Genesis environment.

        Parameters
        ----------
        """
        self.scene: gs.Scene = scene
        self._add_entities()
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
    ) -> list[RenderOutput]:
        """
        Renders the environment.

        Returns
        -------
        image: numpy.ndarray
            The rendered image.
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
            output.append(cam(rgb, depth, segmentation, colorized_seg, normal))

        return output
