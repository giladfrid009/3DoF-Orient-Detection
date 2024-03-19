import numpy as np
import time

from utils.orient import OrientUtils
from algs.algorithm import Algorithm, RunConfig
from view_sampler import ViewSampler
from tqdm.auto import tqdm
from loss_funcs import *


class RandomSampling(Algorithm):
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

        rng = np.random.default_rng(run_config.seed)

        low = np.expand_dims(OrientUtils.LOWER_BOUND, 0)
        high = np.expand_dims(OrientUtils.UPPER_BOUND, 0)

        rnd_orients = rng.uniform(low=low, high=high, size=(self.num_samples, 3)).tolist()

        tqdm_bar = tqdm(
            iterable=rnd_orients,
            disable=run_config.silent,
            leave=False,
        )

        start_time = time.time()

        for test_orient in tqdm_bar:
            loss = self.calc_loss(ref_location, ref_img, test_orient)

            if loss < lowest_loss:
                lowest_loss = loss
                best_orient = test_orient

            tqdm_bar.set_postfix_str(f"Loss: {lowest_loss:.5f}")

            if time.time() - start_time > run_config.time_limit:
                break

        tqdm_bar.close()

        return best_orient, lowest_loss
