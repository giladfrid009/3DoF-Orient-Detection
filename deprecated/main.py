from typing import Iterable
import numpy as np

from view_sampler import *
from algs import *

import loss_funcs
from evaluate import eval_funcs

from evaluate.dataset import Dataset
from evaluate.eval_log import EvalLog
from evaluate.evaluator import Evaluator
from evaluate.dataset import Dataset
from utils.concurrent import TqdmPool

import mealpy

OBJ_LOCATION = (0, 1.3, 0.3)


EVAL_PENALTY = {
    "airplane": 0.08,
    "hammer": 0.11,
    "hand": 0.06,
    "headphones": 0.12,
    "mouse": 0.06,
    "mug": 0.06,
    "stapler": 0.05,
    "toothpaste": 0.06,
}

def evaluate(
    optimizer: mealpy.Optimizer,
    run_config: RunConfig,
    obj_name: str,
    eval_positions: Iterable[ObjectPosition],
    log_folder: str,
):
    cam_config = CameraConfig(location=(0, 0, 0.3), rotation=(np.pi / 2, 0, 0), fov=60)

    with ViewSampler(f"data/{obj_name}/world.xml", cam_config) as world_viewer, ViewSampler(
        f"data/{obj_name}/world_sim.xml", cam_config
    ) as sim_viewer:

        alg = MealAlgorithm(sim_viewer, loss_funcs.IOU(), optimizer)
        log = EvalLog(alg)

        eval_func = eval_funcs.XorDiff(EVAL_PENALTY[obj_name])
        evaluator = Evaluator(world_viewer, sim_viewer, eval_func=eval_func, silent=True)
        evaluator.evaluate(alg, run_config, eval_positions, log)

        log.save(log_folder)


