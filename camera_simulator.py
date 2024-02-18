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
        #self.manipulated_object.zero_velocities()

        self.renderer = mj.Renderer(self.model, resolution[0], resolution[1])
        self.depth_renderer = mj.Renderer(self.model, resolution[0], resolution[1])
        self.depth_renderer.enable_depth_rendering()

    def set_object_position(self, position):
        self.manipulated_object.set_position(position)

    def set_object_orientation_euler(self, orientation):
        self.manipulated_object.set_orientation_euler(orientation)

    def render(self, rotation_matrix, position):
        mj.mj_forward(self.model, self.data)
        self.data.cam_xpos = position
        self.data.cam_xmat = rotation_matrix.flatten()
        self.renderer.update_scene(self.data, camera=0)
        return self.renderer.render()

    def render_depth(self, rotation_matrix, position):
        mj.mj_forward(self.model, self.data)
        self.data.cam_xpos = position
        self.data.cam_xmat = rotation_matrix.flatten()
        self.depth_renderer.update_scene(self.data, camera=0)
        return self.depth_renderer.render()

    def step_simulation(self):
        mj.mj_step(self.model, self.data)
