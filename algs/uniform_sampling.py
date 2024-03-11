import numpy as np
from dataclasses import dataclass
import time

from algs.algorithm import Algorithm, SearchConfig
from view_sampler import ViewSampler
from tqdm.auto import tqdm
from metric_funcs import *


class UniformSampling(Algorithm):

    @dataclass
    class Config(SearchConfig):
        max_samples: int = 1000
        rnd_seed: int = None
        silent: bool = False

    def __init__(self, test_viewer: ViewSampler, metric_func: MetricFunc):
        super().__init__(test_viewer, metric_func)

    def find_orientation(
        self,
        ref_img: np.ndarray,
        ref_position: tuple[float, float, float],
        config: Config,
    ) -> tuple[float, float, float]:

        n = int((config.max_samples / 4) ** (1 / 3))

        alpha = np.linspace(0, 2 * np.pi, 2 * n)
        beta = np.linspace(0, np.pi, n)
        gamma = np.linspace(0, 2 * np.pi, 2 * n)
        orient_grid = np.vstack(np.meshgrid(alpha, beta, gamma, indexing="xy")).reshape(3, -1).T

        # shuffle the orientations
        """ rnd_generator = np.random.default_rng(config.rnd_seed)
        idx = rnd_generator.random(size=orient_grid.shape).argsort(axis=0)
        orient_grid = np.take_along_axis(orient_grid, idx, axis=0) """

        test_orientations = tqdm(iterable=orient_grid.tolist(), disable=config.silent, leave=False)

        start_time = time.time()
        lowest_loss = np.inf
        best_orient = None

        for test_orient in test_orientations:
            loss = self.calc_loss(ref_position, ref_img, test_orient)

            if loss < lowest_loss:
                lowest_loss = loss
                best_orient = test_orient

            test_orientations.set_postfix_str(f"Loss: {lowest_loss:.2f}")

            if time.time() - start_time > config.time_limit:
                break

        return best_orient
