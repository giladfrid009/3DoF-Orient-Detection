import numpy as np
from dataclasses import dataclass
import time

from algs.algorithm import Algorithm, SearchConfig
from view_sampler import ViewSampler
from tqdm.auto import tqdm
from metric_funcs import *


class RandomSampling(Algorithm):

    @dataclass
    class Config(SearchConfig):
        num_samples: int = 1000
        silent: bool = False

    def __init__(self, test_viewer: ViewSampler, metric_func: MetricFunc):
        super().__init__(test_viewer, metric_func)

    def find_orientation(
        self,
        ref_img: np.ndarray,
        ref_position: tuple[float, float, float],
        config: Config,
    ) -> tuple[float, float, float]:
        lowest_loss = np.inf
        best_orient = None

        rnd_orients = np.random.uniform(0, 2 * np.pi, size=(config.num_samples, 3)).tolist()

        start_time = time.time()

        test_orientations = tqdm(
            iterable=rnd_orients,
            disable=config.silent,
        )

        for test_orient in test_orientations:
            loss = self.calc_loss(ref_position, ref_img, test_orient)

            if loss < lowest_loss:
                lowest_loss = loss
                best_orient = test_orient

            test_orientations.set_postfix_str(f"Lowest loss: {lowest_loss:.2f}")

            if time.time() - start_time > config.time_limit:
                break

        return best_orient
