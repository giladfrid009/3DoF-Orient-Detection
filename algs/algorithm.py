import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

from view_sampler import ViewSampler
from manipulated_object import ObjectPosition
from loss_funcs import LossFunc
from image_helpers import ImageHelpers


@dataclass
class SearchConfig:
    time_limit: float = 100
    silent = False


class Algorithm(ABC):
    def __init__(self, test_viewer: ViewSampler, loss_func: LossFunc):
        self._test_viewer = test_viewer
        self.loss_func = loss_func
        self._callback_funcs = []

    def register_loss_callback(self, callback: Callable[[tuple[float, float, float], float], None]):
        self._callback_funcs.append(callback)

    def calc_loss(
        self,
        ref_location: tuple[float, float, float],
        ref_img: np.ndarray,
        test_orient: tuple[float, float, float],
    ) -> float:
        test_img, _ = self._test_viewer.get_view_cropped(ObjectPosition(test_orient, ref_location))

        pad_shape = np.maximum(ref_img.shape, test_img.shape)
        ref_img = ImageHelpers.pad_to_shape(ref_img, pad_shape)
        test_img = ImageHelpers.pad_to_shape(test_img, pad_shape)

        loss = self.loss_func(ref_img, test_img)

        for callback in self._callback_funcs:
            callback(test_orient, loss)

        return loss

    @abstractmethod
    def find_orientation(
        self,
        ref_img: np.ndarray,
        ref_location: tuple[float, float, float],
        alg_config: SearchConfig,
    ) -> tuple[tuple[float, float, float], float]:
        pass
