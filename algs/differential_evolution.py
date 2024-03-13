import numpy as np
from dataclasses import dataclass
import time
import sko
from typing import Callable

from algs.algorithm import Algorithm, SearchConfig
from view_sampler import ViewSampler
from tqdm.auto import tqdm
from loss_funcs import *


class TqdmDE(sko.DE.DE):
    def __init__(
        self,
        func: Callable,
        num_iters: int,
        pop: int,
        mut_prob: float,
        F: float,
        silent: bool = False,
    ):
        """
        Parameters
        ----------------
        func : function
            The func you want to do optimal
        size_pop : int
            Size of population
        num_iters : int
            Number of iters
        prob_mut : float between 0 and 1
            Probability of mutation
        """

        super().__init__(
            func=func,
            n_dim=3,
            lb=[0, 0, 0],
            ub=[2 * np.pi, 2 * np.pi, 2 * np.pi],
            F=F,
            size_pop=pop,
            max_iter=num_iters,
            prob_mut=mut_prob,
        )

        self.silent = silent

    def run(self, time_limit: float) -> tuple[tuple[float, float, float], float]:
        start_time = time.time()

        tqdm_bar = tqdm(
            range(self.max_iter),
            disable=self.silent,
            leave=False,
        )

        self.best_x = None
        self.best_y = np.inf

        for _ in tqdm_bar:
            self.mutation()
            self.crossover()
            self.selection()

            # record the best ones
            generation_best_index = self.Y.argmin()
            self.generation_best_X.append(self.X[generation_best_index, :].copy())
            self.generation_best_Y.append(self.Y[generation_best_index])
            self.all_history_Y.append(self.Y)

            # update global best
            if self.best_y > self.generation_best_Y[-1]:
                self.best_x = self.generation_best_X[-1]
                self.best_y = self.generation_best_Y[-1]

            tqdm_bar.set_postfix_str(f"Loss: {self.best_y:.5f}")

            if time.time() - start_time > time_limit:
                break

        tqdm_bar.close()

        return self.best_x, self.best_y


class DifferentialEvolution(Algorithm):

    @dataclass
    class Config(SearchConfig):
        num_iters: int = 100
        population: int = 50
        mut_prob: float = 0.3
        F: float = 0.5
        silent: bool = False

    def __init__(self, test_viewer: ViewSampler, loss_func: LossFunc):
        super().__init__(test_viewer, loss_func)

    def find_orientation(
        self,
        ref_img: np.ndarray,
        ref_location: tuple[float, float, float],
        alg_config: Config,
    ) -> tuple[tuple[float, float, float], float]:

        func = lambda test_orient: self.calc_loss(ref_location, ref_img, test_orient)

        alg = TqdmDE(
            func,
            num_iters=alg_config.num_iters,
            pop=alg_config.population,
            mut_prob=alg_config.mut_prob,
            F=alg_config.F,
            silent=alg_config.silent,
        )

        orient, loss = alg.run(time_limit=alg_config.time_limit)

        return orient, loss
