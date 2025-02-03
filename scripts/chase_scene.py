import os

# os.environ["PYOPENGL_PLATFORM"] = "glx"
os.environ["MUJOCO_GL"] = "egl"
import genesis as gs
from genesis.engine.entities import RigidEntity

gs.init(backend=gs.gpu)

scene = gs.Scene(
    show_viewer=False,
    viewer_options=gs.options.ViewerOptions(
        res=(1280, 960),
        camera_pos=(3.5, 0.0, 2.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=40,
        max_FPS=60,
    ),
    vis_options=gs.options.VisOptions(
        show_world_frame=True,
        world_frame_size=1.0,
        show_link_frame=False,
        show_cameras=False,
        plane_reflection=True,
        ambient_light=(0.1, 0.1, 0.1),
    ),
    renderer=gs.renderers.Rasterizer(),
)

plane: RigidEntity = scene.add_entity(
    gs.morphs.Plane(),
)
adv: RigidEntity = scene.add_entity(
    gs.morphs.MJCF(file="genesis_gymnasium/envs/assets/xmls/agents/point.xml"),
)
robot: RigidEntity = scene.add_entity(
    gs.morphs.MJCF(
        file="genesis_gymnasium/envs/assets/xmls/agents/car.xml", pos=(0.5, 0, 0.1)
    ),
)

cam = scene.add_camera(
    res=(640, 480),
    pos=(4.5, 0.0, 3.5),
    lookat=(0, 0, 0.5),
    fov=30,
    GUI=False,
)

scene.build()

# render rgb, depth, segmentation, and normal
# rgb, depth, segmentation, normal = cam.render(rgb=True, depth=True, segmentation=True, normal=True)

cam.start_recording()
import numpy as np

jnt_names: list[str] = ["x", "z"]
rob_jnt_names: list[str] = ["left_joint", "right_joint"]
# joint_idx: list = [adv.get_joint(name).dof_idx_local for name in jnt_names]
rob_jnt_idx: list = [robot.get_joint(name).dof_idx_local for name in rob_jnt_names]

for i in range(120):
    adv.control_dofs_velocity([0.5, 0.1, 0], dofs_idx_local=[0, 1, 2])
    robot.control_dofs_force([1, 1], dofs_idx_local=rob_jnt_idx)
    print("Controller force: ", robot.get_dofs_control_force([0, 1]))
    print("Internal force: ", robot.get_dofs_force([0, 1]))
    scene.step()
    cam.set_pose(
        pos=(4 * np.sin(i / 60), 4 * np.cos(i / 60), 3.5),
        lookat=(0, 0, 0.5),
    )
    cam.render()
cam.stop_recording(save_to_filename="scripts/out/video.mp4", fps=60)
