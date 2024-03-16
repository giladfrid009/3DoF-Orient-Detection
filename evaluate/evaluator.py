from typing import Callable
from tqdm.auto import tqdm
from typing import Iterable
import numpy as np
from pathlib import Path
import time

from utils.image import ImageUtils
from view_sampler import ViewSampler
from evaluate.eval_funcs import EvalFunc
from manipulated_object import ObjectPosition
from algs.algorithm import Algorithm, SearchConfig
from algs.mealpy_algorithm import MealAlgorithm
from evaluate.mealpy_log import MealLog


class Evaluator:
    def __init__(self, world_viewer: ViewSampler, eval_func: EvalFunc) -> None:
        self.world_viewer = world_viewer
        self.eval_func = eval_func

        self.log_enable = False
        self.log_folder = ""

        self._callback_funcs = []

    def register_callback(self, callback: Callable[[float], None]):
        self._callback_funcs.append(callback)
    
    def evaluate(
        self,
        alg: Algorithm,
        alg_config: SearchConfig,
        eval_positions: Iterable[ObjectPosition],
    ) -> list[float]:
        print(f"Evaluating Algorithm: {alg.get_name()}")
        print(f"Alg Config: {alg_config}")
        print(f"Loss Function: {type(alg.loss_func).__name__}")

        losses = []
        alg.set_mode(eval=True)

        create_log = self.log_enable and isinstance(alg, MealAlgorithm)

        if create_log:
            log = MealLog(alg)

        for position in tqdm(eval_positions, desc=f"Evaluating: "):

            ref_img, _ = self.world_viewer.get_view_cropped(
                position,
                depth=False,
                allow_simulation=False,
            )

            pred_orient, _ = alg.solve(ref_img, position.location, alg_config)

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
            ref_depth = ImageUtils.pad_to_shape(ref_depth, pad_shape)
            pred_depth = ImageUtils.pad_to_shape(pred_depth, pad_shape)

            # TODO: NEED TO ZERO OUT OUTLIER BACKGROUND DEPTHS. Remove hardcoded values.
            ref_depth[ref_depth > 20] = 0
            pred_depth[pred_depth > 20] = 0

            loss = self.eval_func(ref_depth, pred_depth)
            losses.append(loss)

            if create_log:
                log.add_result(loss, position, ObjectPosition(pred_orient, position.location), alg.history)

            for callback in self._callback_funcs:
                callback(loss)

        alg.set_mode(eval=False)

        if create_log:
            self.save_log(log, alg)

        return losses

    def enable_logging(self, log_folder: str):
        self.log_enable = True
        self.log_folder = log_folder

    def disable_logging(self):
        self.log_enable = False
        self.log_folder = None

    def save_log(self, log: MealLog, alg: MealAlgorithm):
        time_str = time.strftime("%Y%m%d-%H%M%S")
        alg_str = alg.get_name()
        save_path = Path(self.log_folder) / f"{alg_str}_{time_str}.pickle"
        log.save(save_path)
