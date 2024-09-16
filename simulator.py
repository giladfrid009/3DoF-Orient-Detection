import mujoco as mj
import numpy as np

from manipulated_object import ManipulatedObject, ObjectPosition


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

        self._cam_pos: np.ndarray = None
        self._cam_xmat: np.ndarray = None

    def set_object_location(self, obj_loc: tuple[float, float, float]):
        self._object.set_location(obj_loc)

    def set_object_orientation(self, orientation):
        self._object.set_orientation(orientation)

    def get_object_orientation(self) -> tuple[float, float, float]:
        return self._object.get_orientation()

    def get_object_position(self) -> ObjectPosition:
        return ObjectPosition.from_object(self._object)

    def align_camera(self, cam_dist: float):
        obj_position = self._object.get_location()
        self._cam_pos = np.array(obj_position) + np.array([0, cam_dist, 0])
        self._cam_xmat = self._compute_look_at_matrix(self._cam_pos, obj_position).flatten()

    def _compute_look_at_matrix(self, cam_pos: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
        forward = np.array(target_pos) - np.array(cam_pos)
        forward /= np.linalg.norm(forward)

        right = np.cross(forward, np.array([0, 0, 1]))
        right /= np.linalg.norm(right)

        up = np.cross(right, forward)
        up /= np.linalg.norm(up)

        look_at_matrix = np.eye(3)
        look_at_matrix[0, :] = right
        look_at_matrix[1, :] = up
        look_at_matrix[2, :] = -forward

        return look_at_matrix

    def render(self) -> np.ndarray:
        if self._cam_pos is None or self._cam_xmat is None:
            raise ValueError("Camera position must be set using align_camera before rendering.")

        mj.mj_forward(self._model, self._data)
        self._data.cam_xpos = self._cam_pos
        self._data.cam_xmat = self._cam_xmat

        self._renderer.update_scene(self._data, camera=0)
        image = self._renderer.render()
        return image

    def render_depth(self) -> np.ndarray:
        if self._cam_pos is None or self._cam_xmat is None:
            raise ValueError("Camera position must be set using align_camera before rendering.")

        mj.mj_forward(self._model, self._data)
        self._data.cam_xpos = self._cam_pos
        self._data.cam_xmat = self._cam_xmat

        self._depth_renderer.update_scene(self._data, camera=0)
        image = self._depth_renderer.render()
        return image

    def close(self):
        self._renderer.close()
        self._depth_renderer.close()
