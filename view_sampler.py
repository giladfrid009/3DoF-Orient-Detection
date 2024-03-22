import numpy as np
from dataclasses import dataclass

from utils.image import ImageUtils
from simulator import Simulator
from manipulated_object import ObjectPosition


@dataclass(frozen=True)
class CameraConfig:
    location: tuple[int, int, int]
    rotation: np.ndarray
    resolution: tuple[int, int] = (300, 300)
    fov: int = 45
    zfar: float = 5.0


class ViewSampler:
    def __init__(self, world_file: str, camera_config: CameraConfig):
        self._simulator = Simulator(
            resolution=camera_config.resolution,
            fov=camera_config.fov,
            world_file=world_file,
        )
        self._camera_config = camera_config

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
    ) -> np.ndarray:

        self.simulator.set_object_location(position.location)
        self.simulator.set_object_orientation(position.orientation)
        image = self._render_image(depth=False)

        if depth:
            segm = ImageUtils.calc_mask(image, bg_value=0, orig_dims=False)
            image = self._render_image(depth=True)
            zero_mask = np.expand_dims(~segm, axis=-1) | (image >= self.camera_config.zfar)
            image[zero_mask] = 0

        return image

    def get_view_cropped(
        self,
        position: ObjectPosition,
        depth: bool = False,
        margin_factor: float = 1.2,
    ) -> np.ndarray:

        image = self.get_view(position, depth)
        mask = ImageUtils.calc_mask(image, bg_value=0)
        x1, y1, x2, y2 = ImageUtils.calc_bboxes(mask, margin_factor)
        cropped = image[x1:x2, y1:y2, :]
        return cropped

    def close(self):
        self.simulator.close()
