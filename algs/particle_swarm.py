import numpy as np
from dataclasses import dataclass
import time
import sko
from typing import Callable

from algs.algorithm import Algorithm, SearchConfig
from view_sampler import ViewSampler
from tqdm.auto import tqdm
from loss_funcs import *


class TqdmPSO(sko.PSO.PSO):
    def __init__(
        self,
        func: Callable,
        pop: int = 40,
        num_iters: int = 150,
        inertia: float = 0.8,
        silent: bool = False,
    ):
        """
        Parameters
        --------------------
        func : function
            The func you want to do optimal
        pop : int
            Size of population, which is the number of Particles. We use 'pop' to keep accordance with GA
        num_iters : int
            Number of iterations
        lb : array_like
            The lower bound of every variables of func
        ub : array_like
            The upper bound of every variables of func
        inertia : float
            The inertia of the particle
        """
        super().__init__(
            func=func,
            n_dim=3,
            lb=[0, 0, 0],
            ub=[2 * np.pi, 2 * np.pi, 2 * np.pi],
            pop=pop,
            max_iter=num_iters,
            w=inertia,
            c1=0.5,
            c2=0.5,
            verbose=(not silent),
        )

    def run(self, time_limit: float = None) -> tuple[tuple[float, float, float], float]:
        start_time = time.time()

        tqdm_bar = tqdm(
            range(self.max_iter),
            disable=(not self.verbose),
            leave=False,
        )

        for _ in tqdm_bar:
            self.update_V()
            self.recorder()
            self.update_X()
            self.cal_y()
            self.update_pbest()
            self.update_gbest()

            self.gbest_y_hist.append(self.gbest_y)

            tqdm_bar.set_postfix_str(f"Loss: {self.gbest_y[0]:.5f}")

            if time_limit and time.time() - start_time > time_limit:
                break

        self.best_x, self.best_y = self.gbest_x, self.gbest_y

        tqdm_bar.close()

        return self.gbest_x


class ParticleSwarm(Algorithm):

    @dataclass
    class Config(SearchConfig):
        population: int = 20
        num_iters: int = 150
        inertia: float = 0.8
        silent: bool = False

    def __init__(self, test_viewer: ViewSampler, loss_func: LossFunc):
        super().__init__(test_viewer, loss_func)

    def find_orientation(
        self,
        ref_img: np.ndarray,
        ref_position: tuple[float, float, float],
        config: Config,
    ) -> tuple[float, float, float]:

        func = lambda test_orient: self.calc_loss(ref_position, ref_img, test_orient)

        alg = TqdmPSO(
            func,
            pop=config.population,
            num_iters=config.num_iters,
            inertia=config.inertia,
            silent=config.silent,
        )

        orient = alg.run(time_limit=config.time_limit)

        return orient
