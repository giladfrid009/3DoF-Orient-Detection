import pickle
from pathlib import Path


def save_pickle(file_path: str, obj):
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(file_path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        raise ValueError(f"error saving object to pickle file: {e}")


def load_pickle(file_path: str) -> object:
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"file does not exist: {file_path}")
    except Exception as e:
        raise ValueError(f"error loading object from pickle file: {e}")
