import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation

from manipulated_object import ManipulatedObject, ObjectConfig


class Simulator:
    def __init__(
        self,
        world_file,
        resolution: tuple[int, int] = (500, 500),
        fov: int = 45,
    ):
        self._model = mj.MjModel.from_xml_path(world_file)
        self._data = mj.MjData(self._model)

        self._model.cam_fovy = fov
        self._model.vis.global_.fovy = fov

        self._object = ManipulatedObject(self._model, self._data)
        self._object.set_orientation([0, 0, 0])

        self._renderer = mj.Renderer(self._model, resolution[0], resolution[1])
        self._depth_renderer = mj.Renderer(self._model, resolution[0], resolution[1])
        self._depth_renderer.enable_depth_rendering()

    def set_object_position(self, obj_pos: tuple[float, float, float]):
        self._object.set_position(obj_pos)

    def set_object_orientation(self, orientation):
        self._object.set_orientation(orientation)

    def get_object_orientation(self) -> tuple[float, float, float]:
        return self._object.get_orientation()

    def get_object_config(self) -> ObjectConfig:
        return ObjectConfig.from_object(self._object)

    def render(self, cam_rot: tuple[float, float, float], cam_pos: tuple[float, float, float]) -> np.ndarray:
        mj.mj_forward(self._model, self._data)
        self._data.cam_xpos = cam_pos
        self._data.cam_xmat = Rotation.from_euler("xyz", cam_rot).as_matrix().flatten()

        self._renderer.update_scene(self._data, camera=0)
        image = self._renderer.render()
        return image

    def render_depth(self, cam_rot: tuple[float, float, float], cam_pos: tuple[float, float, float]) -> np.ndarray:
        mj.mj_forward(self._model, self._data)
        self._data.cam_xpos = cam_pos
        self._data.cam_xmat = Rotation.from_euler("xyz", cam_rot).as_matrix().flatten()
        self._depth_renderer.update_scene(self._data, camera=0)
        image = self._depth_renderer.render()
        return image

    def simulate_seconds(self, seconds: float):
        seconds = max(0, seconds)
        iters = int(seconds / self._model.opt.timestep)
        for _ in range(iters):
            mj.mj_step(self._model, self._data)
