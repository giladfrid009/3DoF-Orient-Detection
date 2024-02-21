from dataclasses import dataclass

from view_converter import ViewConverter


class ManipulatedObject:
    """
    represent, query and manually manipulate the manipulated object
    assuming it's body name is 'manipulated_object'
    """

    def __init__(self, mj_model, mj_data):
        self._model = mj_model
        self._data = mj_data
        self._jntadr = mj_model.body("manipulated_object").jntadr[0]

    def set_position(self, position: tuple[float, float, float]):
        assert len(position) == 3
        self._data.qpos[self._jntadr : self._jntadr + 3] = position

    def set_orientation(self, orientation: tuple[float, float, float]):
        assert len(orientation) == 3
        orient_quat = ViewConverter.euler_to_quat(orientation)
        self._data.qpos[self._jntadr + 3 : self._jntadr + 7] = orient_quat

    def get_orientation(self) -> tuple[float, float, float]:
        rotation = ViewConverter.quat_to_euler(self._data.qpos[self._jntadr + 3 : self._jntadr + 7])
        return rotation

    def get_position(self) -> tuple[float, float, float]:
        return self._data.qpos[self._jntadr : self._jntadr + 3]


@dataclass(frozen=True)
class ObjectConfig:
    orientation: tuple[float, float, float]
    position: tuple[float, float, float]

    @staticmethod
    def from_object(obj_state: ManipulatedObject):
        return ObjectConfig(obj_state.get_orientation(), obj_state.get_position())
