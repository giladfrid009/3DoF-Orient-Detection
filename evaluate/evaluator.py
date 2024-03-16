from tqdm.auto import tqdm
from typing import Iterable
import numpy as np

from image_helpers import ImageHelpers
from view_sampler import ViewSampler
from evaluate.eval_funcs import EvalFunc
from manipulated_object import ObjectPosition
from algs.algorithm import Algorithm, SearchConfig
from exp_logging import Experiment
from algs.mealpy_algorithm import MealpyAlgorithm
from pathlib import Path

class Evaluator:
    def __init__(self, world_viewer: ViewSampler, eval_func: EvalFunc) -> None:
        self.world_viewer = world_viewer
        self.eval_func = eval_func
        self.log_enable = False
        self.root = None

    def evaluate(
        self,
        alg: Algorithm,
        alg_config: SearchConfig,
        eval_positions: Iterable[ObjectPosition],
    ) -> list[float]:
        print(f"Evaluating Algorithm: {type(alg).__name__}")
        print(f"Alg Config: {alg_config}")
        print(f"Loss Function: {type(alg.loss_func).__name__}")

        losses = []
        alg.set_mode(eval=True)

        for iter, position in enumerate(tqdm(eval_positions, desc=f"Evaluating: ")):

            ref_img, _ = self.world_viewer.get_view_cropped(
                position,
                depth=False,
                allow_simulation=False,
            )

            pred_orient, _ = alg.find_orientation(ref_img, position.location, alg_config)

            ref_depth, _ = self.world_viewer.get_view_cropped(
                position=position,
                depth=True,
                allow_simulation=False,
            )

            pred_depth, _ = self.world_viewer.get_view_cropped(
                position=ObjectPosition(pred_orient, position.location),
                depth=True,
                allow_simulation=False,
            )

            pad_shape = np.maximum(ref_depth.shape, pred_depth.shape)
            ref_depth = ImageHelpers.pad_to_shape(ref_depth, pad_shape)
            pred_depth = ImageHelpers.pad_to_shape(pred_depth, pad_shape)

            # TODO: NEED TO ZERO OUT OUTLIER BACKGROUND DEPTHS
            ref_depth[ref_depth > 20] = 0
            pred_depth[pred_depth > 20] = 0

            loss = self.eval_func(ref_depth, pred_depth)
            losses.append(loss)

            if self.log_enable:
                path = self.root + f"/{alg.get_name()}_{iter}.res"
                self.log_result(path, alg, loss, position, ObjectPosition(pred_orient, position.location))

        alg.set_mode(eval=False)

        return losses

    def enable_logging(self, root:str, exist_ok:bool=False):
        self.log_enable = True
        self.root = root
        Path(root).mkdir(parents=True, exist_ok=exist_ok)
    
    def disable_logging(self):
        self.log_enable = False
        self.root = None

    def log_result(self, file_path:str, alg:MealpyAlgorithm, loss:float, ref_pos:ObjectPosition, pred_pos:ObjectPosition):
        if isinstance(alg, MealpyAlgorithm):
            exp = Experiment(alg.optimizer, loss, ref_pos, pred_pos)
            exp.save(file_path)









