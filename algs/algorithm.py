import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

from view_sampler import ViewSampler
from manipulated_object import ObjectPosition
from loss_funcs import LossFunc
from utils.image import ImageUtils


@dataclass
class RunConfig:
    time_limit: float = 100
    rnd_seed: int = None
    silent: bool = False


class Algorithm(ABC):
    def __init__(self, test_viewer: ViewSampler, loss_func: LossFunc):
        self._test_viewer = test_viewer
        self.loss_func = loss_func
        self.eval_mode = False
        self._callback_funcs = []

    def register_callback(self, callback: Callable[[dict[str, float]], None]):
        self._callback_funcs.append(callback)

    def remove_callback(self, callback: Callable[[dict[str, float]], None]):
        self._callback_funcs.remove(callback)

    def clear_callbacks(self):
        self._callback_funcs.clear()

    def set_mode(self, eval: bool):
        self.eval_mode = eval

    def get_name(self) -> str:
        return type(self).__name__

    def get_params(self) -> dict[str, object]:
        params = dict()
        for name, value in self.__dict__.items():
            if not name.startswith("_") and isinstance(value, (int, float, str, bool)):
                params[name] = value
        return params

    def calc_loss(
        self,
        ref_location: tuple[float, float, float],
        ref_img: np.ndarray,
        test_orient: tuple[float, float, float],
    ) -> float:
        test_img, _ = self._test_viewer.get_view_cropped(
            position=ObjectPosition(test_orient, ref_location),
            allow_simulation=self.eval_mode == False,
        )

        loss = self.loss_func(ref_img, test_img)

        x, y, z = test_orient
        for callback in self._callback_funcs:
            callback(x=x, y=y, z=z, loss=loss)

        return loss

    @abstractmethod
    def solve(
        self,
        ref_img: np.ndarray,
        ref_location: tuple[float, float, float],
        alg_config: RunConfig,
    ) -> tuple[tuple[float, float, float], float]:
        pass
