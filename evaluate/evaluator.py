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
    def __init__(
        self,
        rgb_viewer: ViewSampler,
        depth_viewer: ViewSampler,
        eval_func: EvalFunc,
        silent: bool = False,
    ) -> None:
        self.world_viewer = rgb_viewer
        self.depth_viewer = depth_viewer
        self.eval_func = eval_func
        self.silent = silent

    def evaluate(
        self,
        alg: Algorithm,
        run_config: RunConfig,
        eval_positions: Iterable[ObjectPosition],
        log: EvalLog = None,
        plot: PlotterBase = None,
    ) -> list[float]:

        alg.set_mode(eval=True)

        if not self.silent:
            print(f"Algorithm: {alg.get_name()} | Objective Func: {alg.loss_func.get_name()}")
            print(f"Config: {run_config}")

        if plot is not None:
            alg.register_callback(plot.add_data)

        eval_losses = []

        for position in tqdm(eval_positions, desc=f"Running Evaluation", disable=self.silent):

            if plot is not None:
                plot.reset()

            ref_img, _ = self.world_viewer.get_view_cropped(position, depth=False)
            ref_depth, _ = self.depth_viewer.get_view_cropped(position, depth=True)

            pred_orient, run_hist = alg.solve(ref_img, position.location, run_config)
            pred_position = ObjectPosition(pred_orient, position.location)
            pred_depth, _ = self.depth_viewer.get_view_cropped(pred_position, depth=True)

            eval_loss = self.eval_func(ref_depth, pred_depth)
            eval_losses.append(eval_loss)

            if log is not None:
                log.add_result(eval_loss, run_hist, position, pred_position)

        if plot is not None:
            alg.remove_callback(plot.add_data)

        alg.set_mode(eval=False)

        return eval_losses
