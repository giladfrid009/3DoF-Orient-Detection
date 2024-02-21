import numpy as np
from scipy.spatial.transform import Rotation


class ViewConverter:
    @staticmethod
    def euler_to_quat(orientation: tuple[float, float, float]) -> np.ndarray:
        assert len(orientation) == 3
        return Rotation.from_euler("xyz", orientation, degrees=False).as_quat()

    @staticmethod
    def euler_to_matrix(orientation: tuple[float, float, float]) -> np.ndarray:
        assert len(orientation) == 3
        return Rotation.from_euler("xyz", orientation, degrees=False).as_matrix()

    @staticmethod
    def quat_to_euler(orientation: np.ndarray):
        return Rotation.from_quat(orientation).as_euler("xyz", degrees=False)

    @staticmethod
    def matrix_to_euler(orientation: np.ndarray) -> np.ndarray:
        assert len(orientation) == 3
        return Rotation.from_matrix(orientation).as_euler("xyz", degrees=False)
