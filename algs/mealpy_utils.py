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


