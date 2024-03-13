import numpy as np
from dataclasses import dataclass
import time
import math
from scipy.spatial.transform import Rotation


from algs.algorithm import Algorithm, SearchConfig
from view_sampler import ViewSampler
from tqdm.auto import tqdm
from loss_funcs import *


class UniformSampling(Algorithm):

    @dataclass
    class Config(SearchConfig):
        min_samples: int = 1000
        randomized: bool = False
        rnd_seed: int = None

    def __init__(self, test_viewer: ViewSampler, loss_func: LossFunc):
        super().__init__(test_viewer, loss_func)

    @staticmethod
    def _uniform_det_axes(num_pts: int) -> np.ndarray:
        indices = np.arange(0, num_pts, dtype=float) + 0.5
        phi = np.arccos(1 - 2 * indices / num_pts)
        theta = np.pi * (1 + 5**0.5) * indices
        x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)
        xyz = np.stack((x, y, z), axis=-1)
        return xyz

    @staticmethod
    def _uniform_rnd_axes(num_pts: int, rng: np.random.Generator) -> np.ndarray:
        xyz = rng.normal(size=(num_pts, 3))
        xyz = xyz / np.linalg.norm(xyz, axis=1)[:, np.newaxis]
        return xyz

    def find_orientation(
        self,
        ref_img: np.ndarray,
        ref_location: tuple[float, float, float],
        alg_config: Config,
    ) -> tuple[tuple[float, float, float], float]:
        lowest_loss = np.inf
        best_orient = None

        n = math.ceil(math.sqrt(alg_config.min_samples))
        rng = np.random.default_rng(alg_config.rnd_seed)

        if alg_config.randomized:
            axes = UniformSampling._uniform_rnd_axes(n, rng)
            axes = np.repeat(axes, n, axis=0)
            rots = rng.uniform(0, 2 * np.pi, size=n * n)
        else:
            axes = UniformSampling._uniform_det_axes(n)
            axes = np.repeat(axes, n, axis=0)
            rots = np.linspace(0, 2 * np.pi, num=n, endpoint=False)
            rots = np.tile(rots, n)

        rot_vec = np.expand_dims(rots, axis=-1) * axes
        orients = Rotation.from_rotvec(rot_vec).as_euler("xyz")
        rng.shuffle(orients, axis=0)

        tqdm_bar = tqdm(
            iterable=orients,
            disable=alg_config.silent,
            leave=False,
        )

        start_time = time.time()

        for test_orient in tqdm_bar:
            loss = self.calc_loss(ref_location, ref_img, test_orient)

            if loss < lowest_loss:
                lowest_loss = loss
                best_orient = test_orient

            tqdm_bar.set_postfix_str(f"Loss: {lowest_loss:.5f}")

            if time.time() - start_time > alg_config.time_limit:
                break

        tqdm_bar.close()

        return best_orient, lowest_loss
