import numpy as np
from dataclasses import dataclass

from image_helpers import ImageHelpers
from simulator import Simulator
from manipulated_object import ObjectPosition


@dataclass(frozen=True)
class CameraConfig:
    location: tuple[int, int, int]
    rotation: np.ndarray
    resolution: tuple[int, int] = (300, 300)
    fov: int = 45


class ViewSampler:
    def __init__(
        self,
        world_file: str,
        camera_config: CameraConfig,
        simulation_time: float = 0,
    ):
        self._simulator = Simulator(
            resolution=camera_config.resolution,
            fov=camera_config.fov,
            world_file=world_file,
        )
        self._camera_config = camera_config
        self._simulation_time = simulation_time

    @property
    def simulator(self):
        return self._simulator

    @property
    def camera_config(self) -> CameraConfig:
        return self._camera_config

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _render_image(self, depth: bool):
        if depth:
            image = self.simulator.render_depth(self.camera_config.rotation, self.camera_config.location)
            image = np.expand_dims(image, axis=-1)
            return image
        return self.simulator.render(self.camera_config.rotation, self.camera_config.location)

    def get_view(
        self,
        position: ObjectPosition,
        depth: bool = False,
        allow_simulation: bool = True,
    ) -> tuple[np.ndarray, ObjectPosition]:

        self.simulator.set_object_location(position.location)
        self.simulator.set_object_orientation(position.orientation)
        self.simulator.simulate_seconds(self._simulation_time if allow_simulation else 0)
        image = self._render_image(depth=False)

        if depth:
            mask = ImageHelpers.calc_mask(image, bg_value=0)
            image = self._render_image(depth=True)
            image[~mask] = 0

        position = self.simulator.get_object_position()
        return image, position

    def get_view_cropped(
        self,
        position: ObjectPosition,
        depth: bool = False,
        margin_factor: float = 1.2,
        allow_simulation: bool = True,
    ) -> tuple[np.ndarray, ObjectPosition]:

        image, position = self.get_view(position, depth, allow_simulation=allow_simulation)
        mask = ImageHelpers.calc_mask(image, bg_value=0)
        x1, y1, x2, y2 = ImageHelpers.calc_bboxes(mask, margin_factor)
        cropped = image[x1:x2, y1:y2, :]
        return cropped, position

    def close(self):
        self.simulator.close()
