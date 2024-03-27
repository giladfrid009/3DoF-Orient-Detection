import numpy as np
import mealpy

from algs import *
from view_sampler import ViewSampler, CameraConfig
import loss_funcs
from evaluate.dataset import Dataset

OBJ_LOCATION = (0, 1.3, 0.3)

EVAL_DATASET = Dataset.load("EvalDataset2.pkl")# Dataset.create_random(location=OBJ_LOCATION, num_samples=100, seed=0)
TEST_DATASET = Dataset.load("AllOptimizersDataset.pkl")

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
    mealpy.physics_based.SA.SwarmSA.__name__,
    # mealpy.swarm_based.FFA.OriginalFFA.__name__,
    mealpy.human_based.ICA.OriginalICA.__name__,
    mealpy.swarm_based.SCSO.OriginalSCSO.__name__,
    mealpy.swarm_based.SFO.ImprovedSFO.__name__,
    mealpy.swarm_based.MPA.OriginalMPA.__name__,
    # mealpy.system_based.WCA.OriginalWCA.__name__,
    # mealpy.system_based.AEO.AugmentedAEO.__name__,
    #classics
    mealpy.evolutionary_based.GA.BaseGA.__name__,
    mealpy.evolutionary_based.DE.OriginalDE.__name__,
    mealpy.math_based.HC.OriginalHC.__name__,
    mealpy.swarm_based.PSO.OriginalPSO.__name__,
    UniformSampling.__name__,
]

ALGORITHM_PARAMS = {
    "SwarmSA": {
        't0': [500, 1000, 2000],
        'mutation_rate': [0.1, 0.02],
        'mutation_step_size': [0.1, 0.15]
        },

    "OriginalFFA": {
        'gamma': [0.001, 0.005],
        'alpha': [0.2, 0.05, 0.5],
        },
    "OriginalICA": {
        'empire_count': [7, 10],
        'revolution_prob': [0.4, 0.2],
        'revolution_rate': [0.1, 0.2],
        'revolution_step_size': [0.1, 0.2, 0.05],
        },
    "ImprovedSFO": {
        'pp': [0.1, 0.2, 0.05, 0.5],
        },
    "OriginalWCA": {
        'nsr': [4, 7, 10], 
        'wc': [1.5, 2.0, 2.5], 
        'dmax': [1e-06, 1e-5, 1e-4]
        },
    "BaseGA": {
        'pc': [0.95, 0.98, 0.9], 
        'pm': [0.025, 0.1, 0.01],
        },
    "OriginalDE": {
        'wf': [0.1, 0.2], 
        'cr': [0.9, 0.75], 
        'strategy': range(6)
        },
    "OriginalHC": {
        'neighbour_size': [50, 150, 400, 700, 850], 
        },
    "OriginalPSO": {
        'c1': [2.05, 1],
        'c2': [2.05, 1],
        'w': [0.4, 0.3]
        },
}


def create_algorithm(
    name: str,
    sim_viewer: ViewSampler,
    loss_func=loss_funcs.IOU(),
) -> Algorithm:
    if name == mealpy.PSO.OriginalPSO.__name__:
        return MealAlgorithm(sim_viewer, loss_func, mealpy.PSO.OriginalPSO(c1=2.05, c2=1, w=0.3))
    elif name == mealpy.ICA.OriginalICA.__name__:
        return MealAlgorithm(
            sim_viewer,
            loss_func,
            mealpy.ICA.OriginalICA(
                empire_count=7,
                assimilation_coeff=1.5,
                revolution_prob=0.4,
                revolution_rate=0.1,
                revolution_step_size=0.05,
            ),
        )
    elif name == mealpy.HC.OriginalHC.__name__:
        return MealAlgorithm(sim_viewer, loss_func, mealpy.HC.OriginalHC(neighbour_size=850))
    elif name == mealpy.physics_based.SA.SwarmSA.__name__:
        return MealAlgorithm(sim_viewer, loss_func, mealpy.physics_based.SA.SwarmSA(t0=1000, t1=1, mutation_rate=0.1, mutation_step_size=0.1))
    elif name == mealpy.DE.OriginalDE.__name__:
        return MealAlgorithm(sim_viewer, loss_func, mealpy.DE.OriginalDE(wf=0.2, cr=0.9, strategy=2))
    elif name == mealpy.swarm_based.FFA.OriginalFFA.__name__:
        return MealAlgorithm(sim_viewer, loss_func, mealpy.swarm_based.FFA.OriginalFFA(gamma=0.005, alpha=0.2))
    elif name == mealpy.GA.BaseGA.__name__:
        return MealAlgorithm(sim_viewer, loss_func, mealpy.GA.BaseGA(pc=0.9, pm=0.1))    
    elif name == mealpy.system_based.WCA.OriginalWCA.__name__:
        return MealAlgorithm(sim_viewer, loss_func, mealpy.system_based.WCA.OriginalWCA(nsr=4,wc=2.5,dmax=1e-5))    
    elif name == mealpy.swarm_based.SFO.ImprovedSFO.__name__:
        return MealAlgorithm(sim_viewer, loss_func, mealpy.swarm_based.SFO.ImprovedSFO())
    elif name == mealpy.SCSO.OriginalSCSO.__name__:
        return MealAlgorithm(sim_viewer, loss_func, mealpy.SCSO.OriginalSCSO())
    elif name == mealpy.swarm_based.MPA.OriginalMPA.__name__:
        return MealAlgorithm(sim_viewer, loss_func, mealpy.swarm_based.MPA.OriginalMPA())
    elif name == mealpy.system_based.AEO.AugmentedAEO.__name__:
        return MealAlgorithm(sim_viewer, loss_func, mealpy.system_based.AEO.AugmentedAEO())
    elif name == UniformSampling.__name__:
        return UniformSampling(sim_viewer, loss_func, num_samples=10000, epoch_size=50)
    else:
        raise ValueError(f"Unknown algorithm: {name}")
