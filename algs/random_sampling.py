import numpy as np
import time

from utils.orient import OrientUtils
from algs.algorithm import *
from view_sampler import ViewSampler
from loss_funcs import *


class RandomSampling(Algorithm):
    def __init__(self, test_viewer: ViewSampler, loss_func: LossFunc, epoch_size: int = 50):
        super().__init__(test_viewer, loss_func)
        self.epoch_size = epoch_size

    def solve(
        self,
        ref_img: np.ndarray,
        ref_location: tuple[float, float, float],
        run_config: RunConfig,
    ) -> tuple[tuple[float, float, float], RunHistory]:
        lowest_loss = np.inf
        best_orient = None

        start_time = time.time()
        run_hist = RunHistory()
        epoch_start_time = start_time
        
        for epoch in range(run_config.max_epoch):
            orients = OrientUtils.generate_random(self.epoch_size, run_config.seed)
            for test_orient in orients:
                loss = self.calc_loss(ref_location, ref_img, test_orient)
                if loss < lowest_loss:
                    lowest_loss = loss
                    best_orient = test_orient

            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time
            epoch_start_time = epoch_end_time
            run_hist.add_epoch(epoch_time, lowest_loss)
            if epoch_end_time - start_time > run_config.max_time:
                break

        return best_orient, run_hist
