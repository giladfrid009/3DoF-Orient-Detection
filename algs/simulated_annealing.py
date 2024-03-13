import numpy as np
from dataclasses import dataclass
import time
import sko
from typing import Callable

from algs.algorithm import Algorithm, SearchConfig
from view_sampler import ViewSampler
from tqdm.auto import tqdm
from loss_funcs import *


class TqdmSA(sko.SA.SA):
    def __init__(
        self,
        func: Callable,
        init_temp: float,
        num_iters: int,
        samples_per_temp: int,
        stay_counter: int,
        silent: bool,
    ):
        """
        Parameters
        ----------------
        func : function
            The func you want to do optimal
        init_temp :float
            initial temperature
        num_iters : float
            number of iterations under every temperature
        samples_per_temp : int
            num of iteration under every temperature
        """
        super().__init__(
            func=func,
            x0=[0, 0, 0],
            T_max=init_temp,
            T_min=1e-15,
            L=samples_per_temp,
            max_stay_counter=stay_counter,
        )

        self.silent = silent
        self.iter_num = num_iters

    def run(self, time_limit: float) -> tuple[tuple[float, float, float], float]:
        x_current, y_current = self.best_x, self.best_y
        stay_counter = 0

        tqdm_bar = tqdm(
            range(self.iter_num),
            disable=self.silent,
            leave=False,
        )

        start_time = time.time()

        for _ in tqdm_bar:
            for _ in range(self.L):
                x_new = self.get_new_x(x_current)
                y_new = self.func(x_new)

                df = y_new - y_current
                if df < 0 or np.exp(-df / self.T) > np.random.rand():
                    x_current, y_current = x_new, y_new
                    if y_new < self.best_y:
                        self.best_x, self.best_y = x_new, y_new

            self.iter_cycle += 1
            self.cool_down()
            self.generation_best_Y.append(self.best_y)
            self.generation_best_X.append(self.best_x)

            # if best_y stay for max_stay_counter times, stop iteration
            if self.isclose(self.best_y_history[-1], self.best_y_history[-2]):
                stay_counter += 1
            else:
                stay_counter = 0

            if time.time() - start_time > time_limit:
                break

            if self.T < self.T_min:
                break

            if stay_counter > self.max_stay_counter:
                break

            tqdm_bar.set_postfix_str(f"Loss: {self.best_y:.5f}")

        tqdm_bar.close()

        # TODO: WHATS THE DIFFERENCE BETEEN best_y and gbest_y?
        return self.best_x, self.best_y


class SimulatedAnnealing(Algorithm):

    @dataclass
    class Config(SearchConfig):
        init_temp: float = 10
        num_iters: float = 50
        samples_per_temp: int = 100
        stay_counter: int = 150

    def __init__(self, test_viewer: ViewSampler, loss_func: LossFunc):
        super().__init__(test_viewer, loss_func)

    def find_orientation(
        self,
        ref_img: np.ndarray,
        ref_location: tuple[float, float, float],
        alg_config: Config,
    ) -> tuple[tuple[float, float, float], float]:

        func = lambda x: self.calc_loss(ref_location, ref_img, x)

        alg = TqdmSA(
            func,
            init_temp=alg_config.init_temp,
            num_iters=alg_config.num_iters,
            samples_per_temp=alg_config.samples_per_temp,
            stay_counter=alg_config.stay_counter,
            silent=alg_config.silent,
        )

        orient, loss = alg.run(time_limit=alg_config.time_limit)

        return orient, loss
