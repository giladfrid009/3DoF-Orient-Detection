import numpy as np
import time


from utils.orient import OrientUtils
from algs.algorithm import Algorithm, RunConfig
from view_sampler import ViewSampler
from loss_funcs import *


class UniformSampling(Algorithm):
    def __init__(self, test_viewer: ViewSampler, loss_func: LossFunc, num_samples: int = 1000):
        super().__init__(test_viewer, loss_func)
        self.num_samples = num_samples

    def solve(
        self,
        ref_img: np.ndarray,
        ref_location: tuple[float, float, float],
        run_config: RunConfig,
    ) -> tuple[tuple[float, float, float], float]:
        lowest_loss = np.inf
        best_orient = None

        start_time = time.time()

        orients = OrientUtils.generate_uniform(self.num_samples)
        np.random.default_rng(run_config.seed).shuffle(orients, axis=0)

        for test_orient in orients:
            loss = self.calc_loss(ref_location, ref_img, test_orient)

            if loss < lowest_loss:
                lowest_loss = loss
                best_orient = test_orient

            if time.time() - start_time > run_config.max_time:
                break

        return best_orient, lowest_loss


class IDUniformSampling(Algorithm):
    def __init__(self, test_viewer: ViewSampler, loss_func: LossFunc):
        super().__init__(test_viewer, loss_func)

    def solve(
        self,
        ref_img: np.ndarray,
        ref_location: tuple[float, float, float],
        run_config: RunConfig,
    ) -> tuple[tuple[float, float, float], float]:
        lowest_loss = np.inf
        best_orient = None

        start_time = time.time()

        for epoch in range(run_config.max_epoch):
            orients = OrientUtils.generate_uniform((2 + epoch) ** 3)
            for test_orient in orients:
                loss = self.calc_loss(ref_location, ref_img, test_orient)
                if loss < lowest_loss:
                    lowest_loss = loss
                    best_orient = test_orient

            if time.time() - start_time > run_config.max_time:
                break

            return best_orient, lowest_loss
