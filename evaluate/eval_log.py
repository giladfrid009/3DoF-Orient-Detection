from __future__ import annotations
import time
from pathlib import Path
from mealpy.utils.history import History

from manipulated_object import ObjectPosition
from algs.algorithm import Algorithm
from algs.mealpy_algorithm import MealAlgorithm
from utils.io import save_pickle, load_pickle


class EvalLog:
    def __init__(self, algorithm: Algorithm) -> None:
        self.alg_name = algorithm.get_name()
        self.alg_params = algorithm.get_params()

        self.eval_loss_list = []
        self.obj_position_list = []
        self.pred_position_list = []

    def add_result(
        self,
        eval_loss: float,
        obj_position: ObjectPosition,
        pred_position: ObjectPosition,
        *args,
        **kwargs,
    ):
        self.eval_loss_list.append(eval_loss)
        self.obj_position_list.append(obj_position)
        self.pred_position_list.append(pred_position)

    def save(self, folder_path: str):
        time_str = time.strftime("%Y%m%d-%H%M%S")
        save_path = Path(folder_path) / f"{self.alg_name}_{time_str}.pkl"
        save_pickle(save_path, self)

    @staticmethod
    def load(file_path: str) -> EvalLog:
        return load_pickle(file_path)


class MealLog(EvalLog):
    def __init__(self, algorithm: MealAlgorithm) -> None:
        super().__init__(algorithm)
        self.history_list = []

    def add_result(
        self,
        eval_loss: float,
        obj_position: ObjectPosition,
        pred_position: ObjectPosition,
        meal_history: History,
    ):
        super().add_result(eval_loss, obj_position, pred_position)
        self.history_list.append(meal_history)
