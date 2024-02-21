import numpy as np
from simulator import Simulator, CameraConfig
from manipulated_object import ObjectConfig


class ImageSampler:
    def __init__(
        self,
        world_file: str,
        camera_config: CameraConfig,
        simulation_time: float = 0,
    ):
        self._simulator = Simulator(
            resolution=camera_config.resolution,
            fovy=camera_config.fov,
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

    def _render_image(self):
        if self.camera_config.render_depth:
            return self.simulator.render_depth(self.camera_config.rotation, self.camera_config.position)
        return self.simulator.render(self.camera_config.rotation, self.camera_config.position)

    def get_view(self, config: ObjectConfig) -> tuple[np.ndarray, ObjectConfig]:
        self.simulator.set_object_position(config.position)
        self.simulator.set_object_orientation(config.orientation)
        self.simulator.simulate_seconds(self._simulation_time)
        image = self._render_image()
        config = self.simulator.get_object_config()
        return image, config

    def get_view_cropped(self, config: ObjectConfig, margin_factor: float = 1.2) -> tuple[np.ndarray, ObjectConfig]:
        image, config = self.get_view(config)
        mask = ImageHelpers.calc_mask(image, bg_value=0, orig_dims=False)
        x1, y1, x2, y2 = ImageHelpers.calc_bboxes(mask, margin_factor)
        cropped = image[x1:x2, y1:y2, :]
        return cropped, config


class ImageHelpers:
    """
    Image manipulation functions that work both on batched data and single images.
    Image batch dim is assumed to be [N, W, H, 3]
    Single image dim is assumed to be [W, H, 3]
    """

    @staticmethod
    def calc_mask(images: np.ndarray, bg_value: int = 0, orig_dims: bool = False) -> np.ndarray:
        mask = np.any(images != bg_value, axis=-1)
        if orig_dims:
            mask = np.broadcast_to(np.expand_dims(mask, axis=-1), images.shape)
        return mask

    @staticmethod
    def calc_bboxes(mask_batch: np.ndarray, margin_factor: float = 1.2) -> np.ndarray:
        x = np.any(mask_batch, axis=-1)
        y = np.any(mask_batch, axis=-2)

        def argmin_argmax(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            # find smallest and largest indices
            imin = np.argmax(arr, axis=-1)
            arr = np.flip(arr, axis=-1)
            length = arr.shape[-1]
            imax = length - np.argmax(arr, axis=-1)

            # add margin to the indices
            diff = imax - imin
            margin = (diff * (margin_factor - 1)).astype(imin.dtype)
            imin = imin - margin
            imax = imax + margin

            # make sure we're within bounds
            imin = np.maximum(imin, 0)
            imax = np.minimum(imax, length - 1)
            return imin, imax

        xmin, xmax = argmin_argmax(x)
        ymin, ymax = argmin_argmax(y)

        return np.stack((xmin, ymin, xmax, ymax), axis=-1)

    def pad_to_shape(image: np.ndarray, shape: np.ndarray, pad_value: float = 0) -> np.ndarray:
        pad_dims = []
        for width in np.subtract(shape, image.shape):
            pad_dims.append((width // 2, width // 2 + width % 2))
        padded = np.pad(image, pad_dims, mode="constant", constant_values=pad_value)
        return padded
