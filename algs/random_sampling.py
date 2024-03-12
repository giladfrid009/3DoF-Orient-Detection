import numpy as np
from dataclasses import dataclass
import time

from algs.algorithm import Algorithm, SearchConfig
from view_sampler import ViewSampler
from tqdm.auto import tqdm
from loss_funcs import *


class RandomSampling(Algorithm):

    @dataclass
    class Config(SearchConfig):
        num_samples: int = 1000
        rnd_seed: int = None
        silent: bool = False

    def __init__(self, test_viewer: ViewSampler, loss_func: LossFunc):
        super().__init__(test_viewer, loss_func)

    def find_orientation(
        self,
        ref_img: np.ndarray,
        ref_position: tuple[float, float, float],
        config: Config,
    ) -> tuple[float, float, float]:

        start_time = time.time()

        lowest_loss = np.inf
        best_orient = None

        rnd_generator = np.random.default_rng(config.rnd_seed)

        rnd_orients = rnd_generator.uniform(low=0, high=2 * np.pi, size=(config.num_samples, 3)).tolist()

        tqdm_bar = tqdm(
            iterable=rnd_orients,
            disable=config.silent,
            leave=False,
        )

        for test_orient in tqdm_bar:
            loss = self.calc_loss(ref_position, ref_img, test_orient)

            if loss < lowest_loss:
                lowest_loss = loss
                best_orient = test_orient

            tqdm_bar.set_postfix_str(f"Loss: {lowest_loss:.5f}")

            if config.time_limit and time.time() - start_time > config.time_limit:
                break

        tqdm_bar.close()

        return best_orient
