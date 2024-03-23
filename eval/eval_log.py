from __future__ import annotations
import time
from pathlib import Path

from manipulated_object import ObjectPosition
from algs.algorithm import *
from utils.io import save_pickle, load_pickle


class EvalLog:
    def __init__(self, algorithm: Algorithm) -> None:
        self.alg_name = algorithm.get_name()
        self.alg_params = algorithm.get_params()

        self.run_hist_list: list[RunHistory] = []
        self.eval_loss_list = []
        self.obj_position_list = []
        self.pred_position_list = []

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

    def save(self, folder_path: str) -> str:
        time_str = time.strftime("%Y%m%d-%H%M%S")
        save_path = Path(folder_path) / f"{self.alg_name}_{time_str}.pickle"
        save_pickle(save_path, self)
        return save_path.as_posix()

    @staticmethod
    def load(file_path: str) -> EvalLog:
        return load_pickle(file_path)
