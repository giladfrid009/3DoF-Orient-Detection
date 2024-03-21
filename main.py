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

import mealpy


def evaluate(
    alg_name: mealpy.Optimizer,
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

            alg = config.create_algorithm(alg_name, sim_viewer)
            log = EvalLog(alg)
            eval_func = eval_funcs.XorDiff(config.XORDIFF_PENALTY[obj_name])
            evaluator = Evaluator(world_viewer, sim_viewer, eval_func=eval_func, silent=True)
            evaluator.evaluate(alg, run_config, eval_data, log)
            log.save(log_folder)

    except Exception as e:
        raise Exception(f"Error in {alg_name} for {obj_name}: {e}") from e


if __name__ == "__main__":

    exec = TqdmPool(max_workers=4)

    run_config = MealRunConfig(max_time=15, silent=True, seed=0)

    futures = []

    for alg_name in ["UniformSampling", "IDUniformSampling", "RandomSampling"]:
        for obj_name in config.OBJECT_NAMES:

            future = exec.submit(
                evaluate,
                alg_name=alg_name,
                run_config=run_config,
                obj_name=obj_name,
                eval_data=config.EVAL_DATASET,
                log_folder=f"runs/{obj_name}",
            )

            futures.append(future)

    for future in concurrent.futures.as_completed(futures):
        try:
            future.result()
        except Exception as e:
            print(e)

    exec.shutdown(wait=True)
