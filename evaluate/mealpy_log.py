import pickle
from mealpy.utils.history import History

from manipulated_object import ObjectPosition
from algs.mealpy_algorithm import MealAlgorithm
from utils.io import save_pickle, load_pickle


class MealLog:
    def __init__(self, algorithm: MealAlgorithm) -> None:
        self.alg_name = algorithm.get_name()
        self.alg_params = algorithm.get_params()

        self.eval_loss_list = []
        self.obj_position_list = []
        self.pred_position_list = []
        self.history_list = []

    def add_result(
        self,
        eval_loss: float,
        obj_position: ObjectPosition,
        pred_position: ObjectPosition,
        meal_history: History,
    ):
        self.eval_loss_list.append(eval_loss)
        self.obj_position_list.append(obj_position)
        self.pred_position_list.append(pred_position)
        self.history_list.append(meal_history)

    def save(self, file_path: str):
        save_pickle(file_path, self)

    @staticmethod
    def load(file_path: str):
        return load_pickle(file_path)
