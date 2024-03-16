from dataclasses import dataclass
import mealpy
from mealpy.utils.agent import Agent
import numpy as np

from algs.algorithm import Algorithm, SearchConfig
from view_sampler import ViewSampler
from loss_funcs import LossFunc

from time import perf_counter


class MealpyAlgorithm(Algorithm):
    @dataclass
    class Config(SearchConfig):
        seed: int = None
        run_mode: str = "single"
        n_workers: int = None
        log_dest: str = "console"

    def __init__(self, test_viewer: ViewSampler, loss_func: LossFunc, optimizer: mealpy.Optimizer):
        super().__init__(test_viewer, loss_func)
        self.optimizer = optimizer

    def _get_logging_params(self, alg_config: Config) -> tuple[str, str]:
        if alg_config.silent:
            return None, None

        if alg_config.log_dest == "console":
            return "console", None

        return "file", alg_config.log_dest

    def find_orientation(
        self,
        ref_img: np.ndarray,
        ref_location: tuple[float, float, float],
        alg_config: Config,
    ) -> tuple[tuple[float, float, float], float]:

        obj_func = lambda test_orient: self.calc_loss(ref_location, ref_img, test_orient)

        bounds = mealpy.FloatVar(lb=[0, 0, 0], ub=[2 * np.pi, 2 * np.pi, 2 * np.pi])

        log_to, log_file = self._get_logging_params(alg_config)

        problem = mealpy.Problem(
            obj_func=obj_func,
            bounds=bounds,
            minmax="min",
            log_to=log_to,
            log_file=log_file,
            name="orient_detection",
        )

        termination = mealpy.Termination(max_time=perf_counter() + alg_config.time_limit)

        best: Agent = self.optimizer.solve(
            problem=problem,
            mode=alg_config.run_mode,
            n_workers=alg_config.n_workers,
            termination=termination,
            starting_solutions=None,
            seed=alg_config.seed,
        )

        return best.solution, best.target.fitness
