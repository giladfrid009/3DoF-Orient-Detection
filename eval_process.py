from typing import Iterable
import numpy as np
import concurrent.futures

from view_sampler import *
from algs import *

from evaluate import eval_funcs

from evaluate.eval_log import EvalLog
from evaluate.evaluator import Evaluator
from utils.concurrent import TqdmPool
import config
import loss_funcs
import mealpy


def evaluate(
    alg_name: str,
    run_config: RunConfig,
    obj_name: str,
    eval_data: Iterable[ObjectPosition],
    log_folder: str,
):
    try:

        with (
            ViewSampler(f"data/{obj_name}/world.xml", config.CAMERA_CONFIG) as world_viewer,
            ViewSampler(f"data/{obj_name}/world_sim.xml", config.CAMERA_CONFIG) as sim_viewer,
        ):
            # alg = MealAlgorithm(sim_viewer, loss_funcs.IOU(), mealpy.get_optimizer_by_name(alg_name)())
            alg = config.create_algorithm(alg_name, sim_viewer)
            log = EvalLog(alg)
            eval_func = eval_funcs.XorDiff(config.XORDIFF_PENALTY[obj_name])
            evaluator = Evaluator(world_viewer, sim_viewer, eval_func=eval_func, silent=True)
            evaluator.evaluate(alg, run_config, eval_data, log)
            log.save(log_folder)

    except Exception as e:
        raise Exception(f"Error in {alg_name} for {obj_name}: {e}") from e

