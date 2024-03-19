from typing import Iterable
import numpy as np
from itertools import product

from view_sampler import *
from algs import *

import loss_funcs
from evaluate import eval_funcs

from evaluate.dataset import Dataset
from evaluate.eval_log import MealLog
from evaluate.evaluator import Evaluator
from evaluate.dataset import Dataset
from utils.visualize import *

import mealpy

from concurrent.futures import ProcessPoolExecutor

OBJECT_NAMES = ["airplane", "hammer", "hand", "headphones", "mouse", "mug", "stapler", "toothpaste"]

PARAMS: dict[str, dict[str, list]] = {
    mealpy.swarm_based.PSO.OriginalPSO.__name__: {},
    mealpy.swarm_based.MSA.OriginalMSA.__name__: {},
    mealpy.swarm_based.SCSO.OriginalSCSO.__name__: {},
    mealpy.physics_based.SA.OriginalSA.__name__: {},
    mealpy.physics_based.EVO.OriginalEVO.__name__: {},
    mealpy.physics_based.EFO.DevEFO.__name__: {},
    mealpy.physics_based.EO.ModifiedEO.__name__: {},
    mealpy.human_based.ICA.OriginalICA.__name__: {"revolution_prob": [0.4, 0.5, 0.6], "empire_count": [7, 6, 5]},
    mealpy.human_based.FBIO.DevFBIO.__name__: {},
    mealpy.human_based.SARO.OriginalSARO.__name__: {},
    mealpy.evolutionary_based.GA.BaseGA.__name__: {},
    mealpy.evolutionary_based.CRO.OCRO.__name__: {},
    mealpy.evolutionary_based.DE.OriginalDE.__name__: {},
    mealpy.math_based.PSS.OriginalPSS.__name__: {},
    mealpy.math_based.SCA.DevSCA.__name__: {},
    mealpy.math_based.HC.OriginalHC.__name__: {},
}


OPTIMIZERS = [
    mealpy.swarm_based.PSO.OriginalPSO,
    mealpy.swarm_based.MSA.OriginalMSA,
    mealpy.swarm_based.SCSO.OriginalSCSO,
    mealpy.physics_based.SA.OriginalSA,
    mealpy.physics_based.EVO.OriginalEVO,
    mealpy.physics_based.EFO.DevEFO,
    mealpy.physics_based.EO.ModifiedEO,
    mealpy.human_based.ICA.OriginalICA,
    mealpy.human_based.FBIO.DevFBIO,
    mealpy.human_based.SARO.OriginalSARO,
    mealpy.evolutionary_based.GA.BaseGA,
    mealpy.evolutionary_based.CRO.OCRO,
    mealpy.evolutionary_based.DE.OriginalDE,
    mealpy.math_based.PSS.OriginalPSS,
    mealpy.math_based.SCA.DevSCA,
    mealpy.math_based.HC.OriginalHC,
]


def evaluate(
    optimizer: mealpy.Optimizer,
    run_config: RunConfig,
    obj_name: str,
    eval_positions: Iterable[ObjectPosition],
    log_folder: str,
):
    cam_config = CameraConfig(location=(0, 0, 0.3), rotation=(np.pi / 2, 0, 0), fov=60)
    world_viewer = ViewSampler(f"data/{obj_name}/world.xml", cam_config, simulation_time=0)
    sim_viewer = ViewSampler(f"data/{obj_name}/world_sim.xml", cam_config)

    alg = MealAlgorithm(sim_viewer, loss_funcs.IOU(), optimizer)

    log = MealLog(alg)

    evaluator = Evaluator(world_viewer, sim_viewer, eval_func=eval_funcs.XorDiff(0.1))
    evaluator.evaluate(alg, run_config, eval_positions, log)

    log.save(log_folder)

    world_viewer.close()
    sim_viewer.close()


if __name__ == "__main__":
    # exec = ProcessPoolExecutor(max_workers=4, mp_context="spawn")

    run_config = MealRunConfig(time_limit=15, silent=True, seed=0)

    dataset = Dataset.create_random(location=(0, 1.3, 0.3), num_samples=1, seed=1)

    print(PARAMS.keys())

    for optimizer_type in OPTIMIZERS:

        optimizer_params = PARAMS[optimizer_type.__name__]

        for param_config in product(*optimizer_params.values()):
            if len(param_config) == 0:
                continue

            kwargs = {}
            for i, param_name in enumerate(optimizer_params.keys()):
                kwargs[param_name] = param_config[i]

            optimizer = optimizer_type(**kwargs)

            for obj_name in OBJECT_NAMES:
                evaluate(
                    optimizer=optimizer,
                    run_config=run_config,
                    obj_name=obj_name,
                    eval_positions=dataset,
                    log_folder=f"grid_search/{obj_name}",
                )