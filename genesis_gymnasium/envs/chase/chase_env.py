import genesis as gs

from genesis_gymnasium.core.genesis_env import GenesisEnv, DEFAULT_SCENE, RenderMode


class ChaseEnv(GenesisEnv):
    def __init__(
        self, scene: gs.Scene = DEFAULT_SCENE, render_mode: RenderMode = "rgb_array"
    ) -> None:
        super().__init__(scene=scene, render_mode=render_mode)

    def _add_entities(self) -> None:
        self.plane = self.scene.add_entity(gs.morphs.Plane())
        self.agent = self.scene.add_entity(
            gs.morphs.MJCF(file="genesis_gymnasium/envs/assets/xmls/agents/car.xml")
        )
        # self.adversary = self.scene.add_entity(
        #     gs.morphs.MJCF(file="genesis_gymnasium/envs/assets/xmls/agents/point.xml")
        # )

    def _add_camera(self) -> None:
        self.scene.add_camera(
            res=(1280, 960), pos=(3.5, 0.0, 2.5), lookat=(0, 0, 0.5), fov=30, GUI=False
        )
