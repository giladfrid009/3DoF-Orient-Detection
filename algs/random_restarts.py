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
        inner_config: SearchConfig
        num_restarts: int = 5

    def __init__(self, inner_alg: Algorithm):
        super().__init__(None, None)
        self.inner_alg = inner_alg

    def calc_loss(
        self,
        ref_position: tuple[float, float, float],
        ref_img: np.ndarray,
        test_orient: tuple[float, float, float],
    ) -> float:
        raise NotImplementedError("RandomRestarts does not support calc_loss")

    def find_orientation(
        self,
        ref_img: np.ndarray,
        ref_position: tuple[float, float, float],
        alg_config: Config,
    ) -> tuple[float, float, float]:

        inner_config = copy.deepcopy(alg_config.inner_config)

        lowest_loss = np.inf
        best_orient = None

        tqdm_bar = tqdm(
            iterable=range(alg_config.num_restarts),
            disable=inner_config.silent,
            leave=False,
        )

        start_time = time.time()

        for _ in tqdm_bar:

            # we update the time limit for the inner algorithm
            inner_config.time_limit = min(inner_config.time_limit, alg_config.time_limit - (time.time() - start_time))

            orient, loss = self.inner_alg.find_orientation(ref_img, ref_position, inner_config)

            if loss < lowest_loss:
                lowest_loss = loss
                best_orient = orient

            if time.time() - start_time > alg_config.time_limit:
                break

        tqdm_bar.close()

        return best_orient, lowest_loss
