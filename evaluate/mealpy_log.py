import pickle
from mealpy.utils.history import History
from pathlib import Path

from manipulated_object import ObjectPosition
from algs.mealpy_algorithm import MealAlgorithm
import pandas as pd

class MealLog:
    def __init__(self, algorithm: MealAlgorithm) -> None:
        self.alg_name = algorithm.get_name()
        self.alg_params = algorithm.get_params()

        self.eval_loss_list = []
        self.obj_position_list = []
        self.pred_position_list = []
        self.history_list = []
        self.trajectory:list[list[tuple[int,int,int]]] = []

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

    # hook for optimizer callback
    def add_trajectory(self, x:float, y:float, z:float, loss:float):
        sample = len(self.eval_loss_list)
        self.trajectory[sample].append({"pred_orientation":[x,y,z], "loss":loss})
    
    def to_dataframe(self, add_params:bool=False)->pd.DataFrame:
        n_samples = len(self.eval_loss_list)

        data = {"alg": [ *([self.alg_name]*n_samples) ],
                "sample":[*range(n_samples)],
                "eval_loss":[*self.eval_loss_list],
                "ref_pos": [*self.obj_position_list],
                "pred_pos": [*self.pred_position_list],
                }
        if add_params:
            for param, value in self.alg_params.items():
                data[param] = [*([value]*n_samples)]

        return pd.DataFrame(data)

    def trajectory_dataframe(self, sample_id:int, add_params:bool=False)->pd.DataFrame:
        assert len(self.trajectory) > sample_id
        trajectory = self.trajectory[sample_id]
        n_epochs = len(trajectory)
        ref_ori = self.obj_position_list[sample_id].orientation
        data = {"alg": [ *([self.alg_name]*n_epochs) ],
                "epoch": [*range(n_epochs)],
                "ref_orientation": [*([ref_ori]*n_epochs)],
                "pred_orientation": [*self.trajectory],
                }
        if add_params:
            for param, value in self.alg_params.items():
                data[param] = [*([value]*n_epochs)]

        return pd.DataFrame(data)

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
