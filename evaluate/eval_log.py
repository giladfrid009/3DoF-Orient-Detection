from __future__ import annotations
import time
from pathlib import Path
from mealpy.utils.history import History
import pandas as pd

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
        self.trajectory: list[list[tuple[int, int, int]]] = []

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

    def add_trajectory(self, x: float, y: float, z: float, loss: float):
        sample = len(self.eval_loss_list)
        self.trajectory[sample].append({"pred_orientation": [x, y, z], "loss": loss})

    def to_dataframe(self, add_params: bool = False) -> pd.DataFrame:
        n_samples = len(self.eval_loss_list)

        data = {
            "alg": [*([self.alg_name] * n_samples)],
            "sample": [*range(n_samples)],
            "eval_loss": [*self.eval_loss_list],
            "ref_pos": [*self.obj_position_list],
            "pred_pos": [*self.pred_position_list],
        }
        if add_params:
            for param, value in self.alg_params.items():
                data[param] = [*([value] * n_samples)]

        return pd.DataFrame(data)

    def trajectory_dataframe(self, sample_id: int, add_params: bool = False) -> pd.DataFrame:
        assert len(self.trajectory) > sample_id
        trajectory = self.trajectory[sample_id]
        n_epochs = len(trajectory)
        ref_ori = self.obj_position_list[sample_id].orientation
        data = {
            "alg": [*([self.alg_name] * n_epochs)],
            "epoch": [*range(n_epochs)],
            "ref_orientation": [*([ref_ori] * n_epochs)],
            "pred_orientation": [*self.trajectory],
        }
        if add_params:
            for param, value in self.alg_params.items():
                data[param] = [*([value] * n_epochs)]

        return pd.DataFrame(data)

    def save(self, folder_path: str):
        time_str = time.strftime("%Y%m%d-%H%M%S")
        save_path = Path(folder_path) / f"{self.alg_name}_{time_str}.pickle"
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
