import pickle
from mealpy.utils.history import History
from pathlib import Path

from manipulated_object import ObjectPosition
from algs.mealpy_algorithm import MealAlgorithm


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
        """
        Save an object to a pickle file.

        Args:
            obj: The object to be saved.
            file_path (str): The path to the pickle file.

        Raises:
            ValueError: If there is an error saving the object to the pickle file.
        """

        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(file_path, "wb") as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            raise ValueError(f"error saving object to pickle file: {e}")

    @staticmethod
    def load(file_path: str):
        """
        Load an object from a pickle file.

        Args:
            file_path (str): The path to the pickle file.

        Returns:
            The loaded object.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If there is an error loading the object from the pickle file.
        """
        try:
            with open(file_path, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"file does not exist: {file_path}")
        except Exception as e:
            raise ValueError(f"error loading object from pickle file: {e}")
