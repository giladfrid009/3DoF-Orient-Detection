from __future__ import annotations
import time
from pathlib import Path
import pandas as pd

from manipulated_object import ObjectPosition
from algs.algorithm import *
from algs.mealpy_algorithm import MealAlgorithm
from utils.io import save_pickle, load_pickle


class EvalLog:
    def __init__(self, algorithm: Algorithm) -> None:
        self.alg_name = algorithm.get_name()
        self.alg_params = algorithm.get_params()

        self.run_hist_list: list[RunHistory] = []
        self.eval_loss_list = []
        self.obj_position_list = []
        self.pred_position_list = []
        self.trajectory: list[list[tuple[int, int, int]]] = []

    def add_result(
        self,
        eval_loss: float,
        run_hist: RunHistory,
        obj_position: ObjectPosition,
        pred_position: ObjectPosition,
    ):
        self.eval_loss_list.append(eval_loss)
        self.run_hist_list.append(run_hist)
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
            data["params"] = str(self.alg_params)
            # for param, value in self.alg_params.items():
            #     data[param] = [*([value] * n_samples)]

        return pd.DataFrame(data)

    def history_dataframe(self, add_params=False):
        df_list = []

        for i, hist in enumerate(self.run_hist_list):
            data = {
                "alg": ([self.alg_name] * hist.num_epochs),
                "sample": ([i] * hist.num_epochs),
                "epoch": range(hist.num_epochs),
                "eval_loss": ([self.eval_loss_list[i]] * hist.num_epochs),
                "list_epoch_time": hist.epoch_time_list,
                "list_global_best_fit": hist.objective_loss_list,
            }

            if add_params:
                data["params"] = str(self.alg_params)
            df_list.append(pd.DataFrame(data))

        return pd.concat(df_list, axis=0, ignore_index=True)

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

    def eval_stats_dataframe(self, add_params: bool = False) -> pd.DataFrame:
        eval_losses = np.array(self.eval_loss_list)
        median = np.median(eval_losses)
        mean = np.mean(eval_losses)
        std = np.std(eval_losses)
        min_val = np.min(eval_losses)
        max_val = np.max(eval_losses)

        data = {
            "alg": [self.alg_name],
            "mean": [mean],
            "median": [median],
            "std": [std],
            "min_val": [min_val],
            "max_val": [max_val],
        }
        if add_params:
            data["params"] = str(self.alg_params)
            # for param, value in self.alg_params.items():
            #     data[param] = [value]

        return pd.DataFrame(data)

    def save(self, folder_path: str) -> str:
        time_str = time.strftime("%Y%m%d-%H%M%S")
        save_path = Path(folder_path) / f"{self.alg_name}_{time_str}.pickle"
        save_pickle(save_path, self)
        return save_path.as_posix()

    @staticmethod
    def load(file_path: str) -> EvalLog:
        return load_pickle(file_path)
