from dataclasses import dataclass, field
import mealpy
from mealpy import FloatVar, Problem
import numpy as np

from algs.algorithm import Algorithm, SearchConfig
from view_sampler import ViewSampler
from tqdm.auto import tqdm
from loss_funcs import *

from mealpy.physics_based.CDO import OriginalCDO


@dataclass
class TerminationConfig:
    max_epoch: int = None
    max_fe: int = None  # number of function evaluation
    max_time: int = None
    max_early_stop: int = 30


bounds = FloatVar(lb=[0, 0, 0], ub=[2 * np.pi, 2 * np.pi, 2 * np.pi], name="pos")


@dataclass
class ProblemConfig:
    name: str
    obj_func: None
    bounds: FloatVar = field(default=bounds)
    minmax: str = field(default="min")
    log_to: str = field(default="console")


class Chernobyl(Algorithm):

    @dataclass
    class Config(SearchConfig):
        population: int = 20
        num_iters: int = 150
        inertia: float = 0.8

    def __init__(self, test_viewer: ViewSampler, loss_func: LossFunc):
        super().__init__(test_viewer, loss_func)

    def find_orientation(
        self,
        ref_img: np.ndarray,
        ref_location: tuple[float, float, float],
        alg_config: Config,
    ) -> tuple[tuple[float, float, float], float]:

        func = lambda test_orient: self.calc_loss(ref_location, ref_img, test_orient)

        problem = ProblemConfig("config", obj_func=func)
        termination = TerminationConfig()
        alg = OriginalCDO(epoch=10000, pop_size=50)
        # print(problem.__dict__)
        best = alg.solve(problem.__dict__, termination=termination.__dict__)
        # print(best.target)
        # alg.history.save_runtime_chart()
        return best.solution, best.target.fitness
