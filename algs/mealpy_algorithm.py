from dataclasses import dataclass
import mealpy
from mealpy.utils.agent import Agent
from mealpy.utils.history import History
import numpy as np

from utils.orient import OrientUtils
from algs.algorithm import Algorithm, RunConfig
from view_sampler import ViewSampler
from loss_funcs import LossFunc


class MealTermination(mealpy.Termination):
    def __init__(self, max_time: float):
        super().__init__(max_time=max_time)

    def should_terminate(self, current_epoch, current_fe, current_time, current_threshold):
        # Check maximum number of generations
        if self.max_epoch is not None and current_epoch >= self.max_epoch:
            self.message = (
                "Stopping criterion with maximum number of epochs/generations/iterations (MG) occurred. End program!"
            )
            return True
        # Check maximum number of function evaluations
        if self.max_fe is not None and current_fe >= self.max_fe:
            self.message = "Stopping criterion with maximum number of function evaluations (FE) occurred. End program!"
            return True
        # Check maximum time
        if self.max_time is not None and current_time - self.start_time >= self.max_time:
            self.message = (
                "Stopping criterion with maximum running time/time bound (TB) (seconds) occurred. End program!"
            )
            return True
        # Check early stopping
        if self.max_early_stop is not None and current_threshold >= self.max_early_stop:
            self.message = "Stopping criterion with early stopping (ES) (fitness-based) occurred. End program!"
            return True
        return False


@dataclass
class MealRunConfig(RunConfig):
    """
    Args:
        run_mode: Parallel: 'process', 'thread'; Sequential: 'swarm', 'single'.

            * 'process': The parallel mode with multiple cores run the tasks
            * 'thread': The parallel mode with multiple threads run the tasks
            * 'swarm': The sequential mode that has no effect on updating phase of other agents
            * 'single': The sequential mode that effect on updating phase of other agents, this is default mode

        n_workers: The number of workers (cores or threads) to do the tasks (effect only on parallel mode)
        termination: The termination dictionary or an instance of Termination class
        log_dest: The destination of the logging output, 'console' or explicit file path. If silent is True, this is ignored.
        save_pop: Save the population in history or not. Useful for plotting, but otherwise leave as default.
    """

    run_mode: str = "single"
    n_workers: int = None
    save_pop: bool = None


class MealAlgorithm(Algorithm):
    def __init__(self, test_viewer: ViewSampler, loss_func: LossFunc, optimizer: mealpy.Optimizer):
        super().__init__(test_viewer, loss_func)
        self.optimizer = optimizer

    @property
    def history(self) -> History:
        return self.optimizer.history

    def get_name(self) -> str:
        return self.optimizer.get_name()

    def get_params(self) -> dict:
        return self.optimizer.get_parameters()

    def solve(
        self,
        ref_img: np.ndarray,
        ref_location: tuple[float, float, float],
        run_config: MealRunConfig,
    ) -> tuple[tuple[float, float, float], float]:

        obj_func = lambda test_orient: self.calc_loss(ref_location, ref_img, test_orient)

        bounds = mealpy.FloatVar(lb=OrientUtils.LOWER_BOUND, ub=OrientUtils.UPPER_BOUND)

        problem = mealpy.Problem(
            obj_func=obj_func,
            bounds=bounds,
            minmax="min",
            log_to=None if run_config.silent else "console",
            name="orient_detection",
            save_population=run_config.save_pop,
        )

        termination = MealTermination(run_config.time_limit)

        best: Agent = self.optimizer.solve(
            problem=problem,
            mode=run_config.run_mode,
            n_workers=run_config.n_workers,
            termination=termination,
            starting_solutions=None,
            seed=run_config.seed,
        )

        return best.solution, best.target.fitness
