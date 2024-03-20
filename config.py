import numpy as np
import mealpy

from algs import *
from view_sampler import ViewSampler, CameraConfig
import loss_funcs
from evaluate.dataset import Dataset

OBJ_LOCATION = (0, 1.3, 0.3)

EVAL_DATASET = Dataset.create_random(location=OBJ_LOCATION, num_samples=100, seed=0)

CAMERA_CONFIG = CameraConfig(location=(0, 0, 0.3), rotation=(np.pi / 2, 0, 0), fov=60)

OBJECT_NAMES = [
    "airplane",
    "hammer",
    "hand",
    "headphones",
    "mouse",
    "mug",
    "stapler",
    "toothpaste",
]

XORDIFF_PENALTY = {
    "airplane": 0.08,
    "hammer": 0.11,
    "hand": 0.06,
    "headphones": 0.12,
    "mouse": 0.06,
    "mug": 0.06,
    "stapler": 0.05,
    "toothpaste": 0.06,
}

ALGORITHM_NAMES = [
    mealpy.swarm_based.PSO.OriginalPSO.__name__,
    mealpy.swarm_based.MSA.OriginalMSA.__name__,
    mealpy.swarm_based.SCSO.OriginalSCSO.__name__,
    mealpy.physics_based.SA.OriginalSA.__name__,
    mealpy.physics_based.EVO.OriginalEVO.__name__,
    mealpy.physics_based.EFO.DevEFO.__name__,
    mealpy.physics_based.EO.ModifiedEO.__name__,
    mealpy.human_based.ICA.OriginalICA.__name__,
    mealpy.human_based.FBIO.DevFBIO.__name__,
    mealpy.human_based.SARO.OriginalSARO.__name__,
    mealpy.evolutionary_based.GA.BaseGA.__name__,
    mealpy.evolutionary_based.CRO.OCRO.__name__,
    mealpy.evolutionary_based.DE.OriginalDE.__name__,
    mealpy.math_based.PSS.OriginalPSS.__name__,
    mealpy.math_based.SCA.DevSCA.__name__,
    mealpy.math_based.HC.OriginalHC.__name__,
    UniformSampling.__name__,
    IDUniformSampling.__name__,
    RandomSampling.__name__,
]


def create_algorithm(
    name: str,
    world_viewer: ViewSampler,
    sim_viewer: ViewSampler,
    loss_func=loss_funcs.IOU(),
) -> Algorithm:
    name = name.lower()
    if name == mealpy.PSO.OriginalPSO.__name__.lower():
        return MealAlgorithm(sim_viewer, loss_func, mealpy.PSO.OriginalPSO(c1=1, c2=2.05, w=0.2))
    elif name == mealpy.MSA.OriginalMSA.__name__.lower():
        return MealAlgorithm(sim_viewer, loss_func, mealpy.MSA.OriginalMSA())
    elif name == mealpy.SCSO.OriginalSCSO.__name__.lower():
        return MealAlgorithm(sim_viewer, loss_func, mealpy.SCSO.OriginalSCSO())
    elif name == mealpy.SA.OriginalSA.__name__.lower():
        return MealAlgorithm(sim_viewer, loss_func, mealpy.SA.OriginalSA())
    elif name == mealpy.EVO.OriginalEVO.__name__.lower():
        return MealAlgorithm(sim_viewer, loss_func, mealpy.EVO.OriginalEVO())
    elif name == mealpy.EFO.DevEFO.__name__.lower():
        return MealAlgorithm(sim_viewer, loss_func, mealpy.EFO.DevEFO())
    elif name == mealpy.EO.ModifiedEO.__name__.lower():
        return MealAlgorithm(sim_viewer, loss_func, mealpy.EO.ModifiedEO())
    elif name == mealpy.ICA.OriginalICA.__name__.lower():
        return MealAlgorithm(
            sim_viewer,
            loss_func,
            mealpy.ICA.OriginalICA(
                empire_count=7,
                assimilation_coeff=1.5,
                revolution_prob=0.4,
                revolution_rate=0.05,
                revolution_step_size=0.175,
            ),
        )
    elif name == mealpy.FBIO.DevFBIO.__name__.lower():
        return MealAlgorithm(sim_viewer, loss_func, mealpy.FBIO.DevFBIO())
    elif name == mealpy.SARO.OriginalSARO.__name__.lower():
        return MealAlgorithm(sim_viewer, loss_func, mealpy.SARO.OriginalSARO(se=0.5, mu=5))
    elif name == mealpy.GA.BaseGA.__name__.lower():
        return MealAlgorithm(sim_viewer, loss_func, mealpy.GA.BaseGA())
    elif name == mealpy.CRO.OCRO.__name__.lower():
        return MealAlgorithm(sim_viewer, loss_func, mealpy.CRO.OCRO())
    elif name == mealpy.DE.OriginalDE.__name__.lower():
        return MealAlgorithm(sim_viewer, loss_func, mealpy.DE.OriginalDE(strategy=0))
    elif name == mealpy.PSS.OriginalPSS.__name__.lower():
        return MealAlgorithm(sim_viewer, loss_func, mealpy.PSS.OriginalPSS(acceptance_rate=0.925))
    elif name == mealpy.SCA.DevSCA.__name__.lower():
        return MealAlgorithm(sim_viewer, loss_func, mealpy.SCA.DevSCA())
    elif name == mealpy.HC.OriginalHC.__name__.lower():
        return MealAlgorithm(sim_viewer, loss_func, mealpy.HC.OriginalHC(neighbour_size=200))
    elif name == UniformSampling.__name__.lower():
        return UniformSampling(world_viewer, loss_func, num_samples=10000, epoch_size=50)
    elif name == IDUniformSampling.__name__.lower():
        return IDUniformSampling(world_viewer, loss_func)
    elif name == RandomSampling.__name__.lower():
        return RandomSampling(world_viewer, loss_func, epoch_size=50)
    else:
        raise ValueError(f"Unknown algorithm: {name}")
