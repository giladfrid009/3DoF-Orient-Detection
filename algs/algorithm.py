import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass

from view_sampler import ViewSampler
from manipulated_object import ObjectConfig
from loss_funcs import LossFunc
from image_helpers import ImageHelpers


@dataclass
class SearchConfig:
    time_limit: float = None


class Algorithm(ABC):
    def __init__(self, test_viewer: ViewSampler, loss_func: LossFunc):
        self.test_viewer = test_viewer
        self.loss_func = loss_func

    def calc_loss(
        self,
        ref_position: tuple[float, float, float],
        ref_img: np.ndarray,
        test_orient: tuple[float, float, float],
    ) -> float:
        test_img, _ = self.test_viewer.get_view_cropped(ObjectConfig(test_orient, ref_position))

        pad_shape = np.maximum(ref_img.shape, test_img.shape)
        ref_img = ImageHelpers.pad_to_shape(ref_img, pad_shape)
        test_img = ImageHelpers.pad_to_shape(test_img, pad_shape)

        loss = self.loss_func(ref_img, test_img)
        return loss

    @abstractmethod
    def find_orientation(
        self,
        ref_img: np.ndarray,
        ref_position: tuple[float, float, float],
        config: SearchConfig,
    ) -> tuple[float, float, float]:
        pass
