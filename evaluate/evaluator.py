from tqdm.auto import tqdm
from typing import Iterable
import numpy as np

from image_helpers import ImageHelpers
from view_sampler import ViewSampler
from evaluate.eval_funcs import EvalFunc
from manipulated_object import ObjectPosition
from algs.algorithm import Algorithm, SearchConfig


class Evaluator:
    def __init__(self, world_viewer: ViewSampler, eval_func: EvalFunc) -> None:
        self.world_viewer = world_viewer
        self.eval_func = eval_func

    def evaluate(
        self,
        alg: Algorithm,
        alg_config: SearchConfig,
        eval_positions: Iterable[ObjectPosition],
    ) -> list[float]:
        print(f"Evaluating Algorithm: {type(alg).__name__} | Config: {alg_config}")

        losses = []
        alg.set_mode(eval=True)

        for position in tqdm(eval_positions, desc=f"Evaluating: "):

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

            loss = self.eval_func(ref_depth, pred_depth)
            losses.append(loss)

        alg.set_mode(eval=False)

        return losses
