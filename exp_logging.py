
from dataclasses import dataclass, field, InitVar
import mealpy
from mealpy import FloatVar, Problem, Termination
from mealpy.utils.history import History
import pickle
from manipulated_object import ObjectPosition




@dataclass
class Experiment:
    algorithm:InitVar[mealpy.Optimizer]
    eval_loss:float
    obj_position:ObjectPosition
    pred_position:ObjectPosition
    alg_name:str = field(init=False)
    alg_params:dict = field(init=False)
    # problem:Problem = field(init=False)
    termination_config:Termination = field(init=False)
    history:History = field(init=False)
    
    def __post_init__(self, algorithm:mealpy.Optimizer):
        self.alg_name = algorithm.get_name()
        self.alg_params = algorithm.get_parameters()
        self.termination_config = algorithm.termination.__dict__
        self.history = algorithm.history
        
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





















