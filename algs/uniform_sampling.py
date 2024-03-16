import numpy as np
from dataclasses import dataclass
import time


from utils.orient import OrientUtils
from algs.algorithm import Algorithm, SearchConfig
from view_sampler import ViewSampler
from tqdm.auto import tqdm
from loss_funcs import *


class UniformSampling(Algorithm):

    @dataclass
    class Config(SearchConfig):
        min_samples: int = 1000
        randomized: bool = False

    def __init__(self, test_viewer: ViewSampler, loss_func: LossFunc):
        super().__init__(test_viewer, loss_func)

    def solve(
        self,
        ref_img: np.ndarray,
        ref_location: tuple[float, float, float],
        alg_config: Config,
    ) -> tuple[tuple[float, float, float], float]:
        lowest_loss = np.inf
        best_orient = None

        if alg_config.randomized:
            orients = OrientUtils.generate_random(alg_config.min_samples, alg_config.rnd_seed)
        else:
            orients = OrientUtils.generate_uniform(alg_config.min_samples)
            np.random.default_rng(alg_config.rnd_seed).shuffle(orients, axis=0)

        tqdm_bar = tqdm(
            iterable=orients,
            disable=alg_config.silent,
            leave=False,
        )

        start_time = time.time()

        for test_orient in tqdm_bar:
            loss = self.calc_loss(ref_location, ref_img, test_orient)

            if loss < lowest_loss:
                lowest_loss = loss
                best_orient = test_orient

            tqdm_bar.set_postfix_str(f"Loss: {lowest_loss:.5f}")

            if time.time() - start_time > alg_config.time_limit:
                break

        tqdm_bar.close()

        return best_orient, lowest_loss
