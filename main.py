from typing import Iterable
import numpy as np

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
    cam_config = CameraConfig(location=(0, 0, 0.3), rotation=(np.pi / 2, 0, 0), fov=60)

    with ViewSampler(f"data/{obj_name}/world.xml", cam_config) as world_viewer, ViewSampler(
        f"data/{obj_name}/world_sim.xml", cam_config
    ) as sim_viewer:

        alg = config.create_algorithm(alg_name, world_viewer, sim_viewer)
        log = EvalLog(alg)

        eval_func = eval_funcs.XorDiff(config.XORDIFF_PENALTY[obj_name])
        evaluator = Evaluator(world_viewer, sim_viewer, eval_func=eval_func, silent=True)
        evaluator.evaluate(alg, run_config, eval_data, log)

        log.save(log_folder)


if __name__ == "__main__":

    exec = TqdmPool(max_workers=4)

    run_config = MealRunConfig(max_time=15, silent=True, seed=0)

    for alg_name in ["UniformSampling", "IDUniformSampling", "RandomSampling"]:
        for obj_name in config.OBJECT_NAMES:
              
            task = exec.submit(
                evaluate,
                alg_name=alg_name,
                run_config=run_config,
                obj_name=obj_name,
                eval_data=config.EVAL_DATASET,
                log_folder=f"runs/{obj_name}",
            )

    exec.shutdown(wait=True)
