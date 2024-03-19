from tqdm.auto import tqdm
from typing import Iterable

from utils.visualize import PlotterBase
from view_sampler import ViewSampler
from evaluate.eval_funcs import EvalFunc
from manipulated_object import ObjectPosition
from algs.algorithm import Algorithm, RunConfig
from algs.mealpy_algorithm import MealAlgorithm
from evaluate.eval_log import EvalLog


class Evaluator:
    def __init__(self, rgb_viewer: ViewSampler, depth_viewer: ViewSampler, eval_func: EvalFunc) -> None:
        self.world_viewer = rgb_viewer
        self.depth_viewer = depth_viewer
        self.eval_func = eval_func

    def evaluate(
        self,
        alg: Algorithm,
        run_config: RunConfig,
        eval_positions: Iterable[ObjectPosition],
        log: EvalLog = None,
        plotter: PlotterBase = None,
    ) -> list[float]:

        if log is not None and not isinstance(alg, MealAlgorithm):
            raise ValueError("Log can only be used with MealPy algorithms")

        print(f"Algorithm: {alg.get_name()} | Objective Func: {alg.loss_func.get_name()}")
        print(f"Config: {run_config}")

        if plotter is not None:
            alg.register_callback(plotter.add_data)

        losses = []
        alg.set_mode(eval=True)

        for position in tqdm(eval_positions, desc=f"Running Evaluation"):

            if plotter is not None:
                plotter.reset()

            ref_img, _ = self.world_viewer.get_view_cropped(
                position,
                depth=False,
                allow_simulation=False,
            )

            pred_orient, _ = alg.solve(ref_img, position.location, run_config)

            ref_depth, _ = self.depth_viewer.get_view_cropped(
                position=position,
                depth=True,
                allow_simulation=False,
            )

            pred_depth, _ = self.depth_viewer.get_view_cropped(
                position=ObjectPosition(pred_orient, position.location),
                depth=True,
                allow_simulation=False,
            )

            loss = self.eval_func(ref_depth, pred_depth)
            losses.append(loss)

            if log is not None:
                if isinstance(alg, MealAlgorithm):
                    log.add_result(loss, position, ObjectPosition(pred_orient, position.location), alg.history)
                else:
                    log.add_result(loss, position, ObjectPosition(pred_orient, position.location))

        alg.set_mode(eval=False)

        if plotter is not None:
            alg.remove_callback(plotter.add_data)

        return losses
