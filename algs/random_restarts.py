import numpy as np
from dataclasses import dataclass
import time
import copy

from algs.algorithm import Algorithm, SearchConfig
from tqdm.auto import tqdm
from loss_funcs import *


class RandomRestarts(Algorithm):

    @dataclass
    class Config(SearchConfig):
        inner_config: SearchConfig = None
        num_restarts: int = 5

    def __post_init__(self):
        if self.config.inner_config is None:
            raise ValueError("RandomRestarts requires an inner_config to be set")

    def __init__(self, inner_alg: Algorithm):
        super().__init__(None, None)
        self.inner_alg = inner_alg

    def calc_loss(
        self,
        ref_location: tuple[float, float, float],
        ref_img: np.ndarray,
        test_orient: tuple[float, float, float],
    ) -> float:
        raise NotImplementedError("RandomRestarts does not support calc_loss")

    def find_orientation(
        self,
        ref_img: np.ndarray,
        ref_location: tuple[float, float, float],
        alg_config: Config,
    ) -> tuple[float, float, float]:

        inner_config = copy.deepcopy(alg_config.inner_config)
        inner_config.silent = alg_config.silent

        lowest_loss = np.inf
        best_orient = None

        tqdm_bar = tqdm(
            iterable=range(alg_config.num_restarts),
            disable=inner_config.silent,
            leave=False,
            desc="Running Inner Alg",
        )

        start_time = time.time()

        for _ in tqdm_bar:

            # update the time limit for the inner algorithm
            inner_config.time_limit = min(inner_config.time_limit, alg_config.time_limit - (time.time() - start_time))

            orient, loss = self.inner_alg.find_orientation(ref_img, ref_location, inner_config)

            if loss < lowest_loss:
                lowest_loss = loss
                best_orient = orient

            if time.time() - start_time > alg_config.time_limit:
                break

            tqdm_bar.set_postfix_str(f"Loss: {lowest_loss:.5f}")

        tqdm_bar.close()

        return best_orient, lowest_loss
