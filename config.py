import numpy as np
import mealpy

from algs import *
from view_sampler import ViewSampler, CameraConfig
import loss_funcs
from evaluate.dataset import Dataset

OBJ_LOCATION = (0, 1.3, 0.3)

EVAL_DATASET = Dataset.load("AllOptimizersDataset.pkl")# Dataset.create_random(location=OBJ_LOCATION, num_samples=100, seed=0)

CAMERA_CONFIG = CameraConfig(location=(0, 0, 0.3), rotation=(np.pi / 2, 0, 0), fov=60)

OBJECT_NAMES = ["android", "dino", "hammer", 'mug', 'nescafe', 'screwdriver', 'shoe', 'sofa', 'stack_rings']


LOSS_NAMES = [
    loss_funcs.IOU.__name__,
    loss_funcs.MSE.__name__,
    loss_funcs.RMSE.__name__,
    loss_funcs.NMI.__name__,
    loss_funcs.PSNR.__name__,
    loss_funcs.SSIM.__name__,
    loss_funcs.Hausdorff.__name__,
    loss_funcs.ARE.__name__,
    loss_funcs.VI.__name__,
]


def create_loss_func(name: str) -> loss_funcs.LossFunc:
    if name == loss_funcs.IOU.__name__:
        return loss_funcs.IOU()
    elif name == loss_funcs.MSE.__name__:
        return loss_funcs.MSE()
    elif name == loss_funcs.RMSE.__name__:
        return loss_funcs.RMSE(norm="euclidean")
    elif name == loss_funcs.WeightedSum.__name__:
        return loss_funcs.WeightedSum(loss_funcs.IOU(), loss_funcs.RMSE(norm="euclidean"))
    elif name == loss_funcs.NMI.__name__:
        return loss_funcs.NMI(bins=50)
    elif name == loss_funcs.PSNR.__name__:
        return loss_funcs.PSNR()
    elif name == loss_funcs.SSIM.__name__:
        return loss_funcs.SSIM()
    elif name == loss_funcs.Hausdorff.__name__:
        return loss_funcs.Hausdorff()
    elif name == loss_funcs.ARE.__name__:
        return loss_funcs.ARE()
    elif name == loss_funcs.VI.__name__:
        return loss_funcs.VI()


XORDIFF_PENALTY = {
    'android': 0.18898606, 
    'dino': 0.45665917, 
    'hammer': 0.32699254, 
    'mug': 0.17813788, 
    'nescafe': 0.4472909, 
    'screwdriver': 0.17156763, 
    'shoe': 0.36185992, 
    'sofa': 0.46596476, 
    'stack_rings': 0.38177377
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
    sim_viewer: ViewSampler,
    loss_func=loss_funcs.IOU(),
) -> Algorithm:
    if name == mealpy.PSO.OriginalPSO.__name__:
        return MealAlgorithm(sim_viewer, loss_func, mealpy.PSO.OriginalPSO(c1=1, c2=2.05, w=0.2))
    elif name == mealpy.MSA.OriginalMSA.__name__:
        return MealAlgorithm(sim_viewer, loss_func, mealpy.MSA.OriginalMSA())
    elif name == mealpy.SCSO.OriginalSCSO.__name__:
        return MealAlgorithm(sim_viewer, loss_func, mealpy.SCSO.OriginalSCSO())
    elif name == mealpy.SA.OriginalSA.__name__:
        return MealAlgorithm(sim_viewer, loss_func, mealpy.SA.OriginalSA())
    elif name == mealpy.EVO.OriginalEVO.__name__:
        return MealAlgorithm(sim_viewer, loss_func, mealpy.EVO.OriginalEVO())
    elif name == mealpy.EFO.DevEFO.__name__:
        return MealAlgorithm(sim_viewer, loss_func, mealpy.EFO.DevEFO())
    elif name == mealpy.EO.ModifiedEO.__name__:
        return MealAlgorithm(sim_viewer, loss_func, mealpy.EO.ModifiedEO())
    elif name == mealpy.ICA.OriginalICA.__name__:
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
    elif name == mealpy.FBIO.DevFBIO.__name__:
        return MealAlgorithm(sim_viewer, loss_func, mealpy.FBIO.DevFBIO())
    elif name == mealpy.SARO.OriginalSARO.__name__:
        return MealAlgorithm(sim_viewer, loss_func, mealpy.SARO.OriginalSARO(se=0.5, mu=5))
    elif name == mealpy.GA.BaseGA.__name__:
        return MealAlgorithm(sim_viewer, loss_func, mealpy.GA.BaseGA())
    elif name == mealpy.CRO.OCRO.__name__:
        return MealAlgorithm(sim_viewer, loss_func, mealpy.CRO.OCRO())
    elif name == mealpy.DE.OriginalDE.__name__:
        return MealAlgorithm(sim_viewer, loss_func, mealpy.DE.OriginalDE(strategy=0))
    elif name == mealpy.PSS.OriginalPSS.__name__:
        return MealAlgorithm(sim_viewer, loss_func, mealpy.PSS.OriginalPSS(acceptance_rate=0.925))
    elif name == mealpy.SCA.DevSCA.__name__:
        return MealAlgorithm(sim_viewer, loss_func, mealpy.SCA.DevSCA())
    elif name == mealpy.HC.OriginalHC.__name__:
        return MealAlgorithm(sim_viewer, loss_func, mealpy.HC.OriginalHC(neighbour_size=200))
    elif name == UniformSampling.__name__:
        return UniformSampling(sim_viewer, loss_func, num_samples=600, epoch_size=50)
    elif name == IDUniformSampling.__name__:
        return IDUniformSampling(sim_viewer, loss_func)
    elif name == RandomSampling.__name__:
        return RandomSampling(sim_viewer, loss_func, epoch_size=50)
    else:
        raise ValueError(f"Unknown algorithm: {name}")
