
from dataclasses import dataclass, field, InitVar
import mealpy
from mealpy import FloatVar, Problem, Termination
from mealpy.utils.history import History
import pickle
from manipulated_object import ObjectPosition




@dataclass
class Experiment:
    algorithm:InitVar[mealpy.Optimizer]
    alg_name:str = field(init=False)
    alg_params:dict = field(init=False)
    # problem:Problem = field(init=False)
    # termination_config:Termination = field(init=False)
    eval_loss_list:list[float] = field(init=False)
    obj_positions_list:list[ObjectPosition] = field(init=False)
    pred_position_list:list[ObjectPosition] = field(init=False)
    history_list:list[History] = field(init=False)
    
    def __post_init__(self, algorithm:mealpy.Optimizer):
        self.alg_name = algorithm.get_name()
        self.alg_params = algorithm.get_parameters()
        # self.termination_config = algorithm.termination.__dict__

        self.eval_loss_list = []
        self.obj_positions_list = []
        self.pred_position_list = []
        self.history_list = []

    def add_result(self, eval_loss:float, obj_position:ObjectPosition, pred_position:ObjectPosition, history:History):
        self.eval_loss_list.append(eval_loss)
        self.obj_positions_list.append(obj_position)
        self.pred_position_list.append(pred_position)
        self.history_list.append(history)
        
    def save(self, file_path: str):
        """
        Save an object to a pickle file.

        Args:
            obj: The object to be saved.
            file_path (str): The path to the pickle file.

        Raises:
            ValueError: If there is an error saving the object to the pickle file.
        """
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





















