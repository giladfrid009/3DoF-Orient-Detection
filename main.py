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

OBJECT_NAMES = ["airplane", "hammer", "hand", "headphones", "mouse", "mug", "stapler", "toothpaste"]

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

OPTIMIZERS = [
    mealpy.swarm_based.PSO.OriginalPSO(c1=1, c2=2.05, w=0.2),
    mealpy.swarm_based.MSA.OriginalMSA(),
    mealpy.swarm_based.SCSO.OriginalSCSO(),
    mealpy.physics_based.SA.OriginalSA(),
    mealpy.physics_based.EVO.OriginalEVO(),
    mealpy.physics_based.EFO.DevEFO(),
    mealpy.physics_based.EO.ModifiedEO(),
    mealpy.human_based.ICA.OriginalICA(
        empire_count=7,
        assimilation_coeff=1.5,
        revolution_prob=0.4,
        revolution_rate=0.05,
        revolution_step_size=0.175,
    ),
    mealpy.human_based.FBIO.DevFBIO(),
    mealpy.human_based.SARO.OriginalSARO(se=0.5, mu=5),
    mealpy.evolutionary_based.GA.BaseGA(),
    mealpy.evolutionary_based.CRO.OCRO(),
    mealpy.evolutionary_based.DE.OriginalDE(strategy=0),
    mealpy.math_based.PSS.OriginalPSS(acceptance_rate=0.925),
    mealpy.math_based.SCA.DevSCA(),
    mealpy.math_based.HC.OriginalHC(neighbour_size=200),
]


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

        print(log.to_dataframe(False))


if __name__ == "__main__":

    exec = TqdmPool(max_workers=4)

    run_config = MealRunConfig(max_time=15, silent=True, seed=0)

    dataset = Dataset.create_random(location=OBJ_LOCATION, num_samples=1, seed=1)

    tasks = []

    for optimizer in OPTIMIZERS:
        for obj_name in OBJECT_NAMES:
            """ task = exec.submit(
                evaluate,
                optimizer=optimizer,
                run_config=run_config,
                obj_name=obj_name,
                eval_positions=dataset,
                log_folder=f"grid_search/{obj_name}",
            ) """

            evaluate(
                optimizer=optimizer,
                run_config=run_config,
                obj_name=obj_name,
                eval_positions=dataset,
                log_folder=f"grid_search/{obj_name}",
            )
            

    exec.shutdown(wait=True)
