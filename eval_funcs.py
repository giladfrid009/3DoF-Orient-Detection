from abc import ABC, abstractmethod

from view_sampler import ViewSampler
from manipulated_object import ObjectConfig


class EvalFunction(ABC):
    def __init__(self, ref_viewer: ViewSampler, sim_viewer: ViewSampler):
        self.ref_viewer = ref_viewer
        self.sim_viewer = sim_viewer

    def __call__(self, ref_config: ObjectConfig, test_config: ObjectConfig) -> float:
        return self.calc_loss(ref_config, test_config)

    @abstractmethod
    def calc_loss(self, ref_config: ObjectConfig, test_config: ObjectConfig) -> float:
        #TODO: ViewSampler should have an option to render depth on get_view
        #TODO: ViewSampler.get_view_cropped should be able to deal with depth images 
        pass
