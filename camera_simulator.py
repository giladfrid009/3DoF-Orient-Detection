import mujoco as mj
from manipulated_object import ManipulatedObject


class CameraSimulator:
    def __init__(
        self,
        resolution=(500, 500),
        fovy=45,
        world_file="./data/world_mug.xml",
    ):
        self.model = mj.MjModel.from_xml_path(world_file)
        self.data = mj.MjData(self.model)

        self.model.cam_fovy = fovy
        self.model.vis.global_.fovy = fovy

        self.manipulated_object = ManipulatedObject(self.model, self.data)
        self.manipulated_object.set_orientation_euler([0, 0, 0])

        self.renderer = mj.Renderer(self.model, resolution[0], resolution[1])
        self.depth_renderer = mj.Renderer(self.model, resolution[0], resolution[1])
        self.depth_renderer.enable_depth_rendering()

    def set_object_position(self, obj_pos: list | tuple):
        obj_pos = list(obj_pos)
        self.manipulated_object.set_position(obj_pos)

    def set_obj_orient_euler(self, orientation):
        self.manipulated_object.set_orientation_euler(orientation)

    def render(self, cam_rot, cam_pos):
        mj.mj_forward(self.model, self.data)
        self.data.cam_xpos = cam_pos
        self.data.cam_xmat = cam_rot.flatten()
        self.renderer.update_scene(self.data, camera=0)
        return self.renderer.render()

    def render_depth(self, cam_rot, cam_pos):
        mj.mj_forward(self.model, self.data)
        self.data.cam_xpos = cam_pos
        self.data.cam_xmat = cam_rot.flatten()
        self.depth_renderer.update_scene(self.data, camera=0)
        return self.depth_renderer.render()

    def step_simulation(self):
        mj.mj_step(self.model, self.data)

    def simulate_seconds(self, seconds: float):
        seconds = max(0, seconds)
        for _ in range(seconds // self.model.opt.timestep):
            self.step_simulation()
