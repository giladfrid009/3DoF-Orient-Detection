from dataclasses import dataclass
import mealpy
from mealpy.utils.agent import Agent
import numpy as np
import time
from mealpy import Termination
from algs.algorithm import Algorithm, SearchConfig
from view_sampler import ViewSampler
from loss_funcs import LossFunc



class AlgTermination(Termination):
    def __init__(self, max_epoch=None, max_fe=None, max_time=None, max_early_stop=None, **kwargs):
        super().__init__(max_epoch, max_fe, max_time, max_early_stop, **kwargs)

    def should_terminate(self, current_epoch, current_fe, current_time, current_threshold):
        # Check maximum number of generations
        if self.max_epoch is not None and current_epoch >= self.max_epoch:
            self.message = "Stopping criterion with maximum number of epochs/generations/iterations (MG) occurred. End program!"
            return True
        # Check maximum number of function evaluations
        if self.max_fe is not None and current_fe >= self.max_fe:
            self.message = "Stopping criterion with maximum number of function evaluations (FE) occurred. End program!"
            return True
        # Check maximum time
        if self.max_time is not None and current_time - self.start_time >= self.max_time:
            self.message = "Stopping criterion with maximum running time/time bound (TB) (seconds) occurred. End program!"
            return True
        # Check early stopping
        if self.max_early_stop is not None and current_threshold >= self.max_early_stop:
            self.message = "Stopping criterion with early stopping (ES) (fitness-based) occurred. End program!"
            return True
        return False





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

        bounds = mealpy.FloatVar(lb=[-np.pi, -np.pi/2, np.pi], ub=[np.pi, np.pi/2, np.pi])

        log_to, log_file = self._get_logging_params(alg_config)

        problem = mealpy.Problem(
            obj_func=obj_func,
            bounds=bounds,
            minmax="min",
            log_to=log_to,
            log_file=log_file,
            name="orient_detection",
        )

        termination = AlgTermination(max_time=alg_config.time_limit)

        best: Agent = self.optimizer.solve(
            problem=problem,
            mode=alg_config.run_mode,
            n_workers=alg_config.n_workers,
            termination=termination,
            starting_solutions=None,
            seed=alg_config.seed,
        )

        return best.solution, best.target.fitness
    
    def get_name(self)->str:
        return self.optimizer.get_name()
