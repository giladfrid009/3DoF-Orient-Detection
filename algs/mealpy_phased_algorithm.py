import mealpy
from mealpy.utils.agent import Agent
from mealpy.utils.history import History
import numpy as np
import skimage as ski

from utils.orient import OrientUtils
from algs.algorithm import *
from algs.mealpy_algorithm import MealTermination, MealRunConfig
from view_sampler import ViewSampler
from loss_funcs import LossFunc

# VERY IMPROTANT IS TO READ https://scikit-image.org/docs/stable/auto_examples/registration/plot_register_rotation.html#sphx-glr-auto-examples-registration-plot-register-rotation-py
# TODO: FIND OUT WHICH AXIS IS THE ROTATION AXIS (1 2 OR 3)
# TODO: TEST DIRTY IMPLEMENTATION FIRST
# TODO: FIRST PROBLEM IS THAT THE POV CENTER PIXEL SHOULD BE THE ROTATION AXIS CENTER
# TODO: SECOND PROBLEM IS THAT WE'RE CURRENTLY RETURNING get_view_cropped AND ALSO THE ref_img IS CROPPED AND NOT RECTANGULAR.
# THE REF AND TEST IMAGES SHOULD HAVE THE SAME SHAPE
# TODO: THIRD PROBLEM IS THAT WE NEED TO KNOW THE RADIUS OF THE OBJECT IN THE REFERENCE IMAGE

# AFTER DIRTY IMPL IS FINISHED AND WORKING:
# TODO: IMPLEMENT GENERIC PHASED ALGORITHM DIRECTLY INHERETING FROM Algorithm
# TODO: MAKE ALL ALG CLASSES SUPPORT PHASED_ALGORITHM backend


class MealPhasedAlgorithm(Algorithm):
    def __init__(self, test_viewer: ViewSampler, loss_func: LossFunc, optimizer: mealpy.Optimizer):
        super().__init__(test_viewer, loss_func)
        self.optimizer = optimizer

    @property
    def history(self) -> History:
        return self.optimizer.history

    def get_name(self) -> str:
        return self.optimizer.get_name()

    def get_params(self) -> dict:
        return self.optimizer.get_parameters()

    def calc_invariant_loss(
        self,
        ref_location: tuple[float, float, float],
        ref_img: np.ndarray,
        test_orient: tuple[float, float],
    ):
        test_orient = (0, test_orient[0], test_orient[1])

        test_img = self._test_viewer.get_view_cropped(ObjectPosition(test_orient, ref_location))

        ref_polar = ski.transform.warp_polar(ref_img, channel_axis=-1)
        test_polar = ski.transform.warp_polar(test_img, channel_axis=-1)

        shifts, error, phasediff = ski.registration.phase_cross_correlation(
            ref_polar,
            test_polar,
            upsample_factor=1,
        )

        rot_angle = shifts[0]

        test_orient = (rot_angle, test_orient[1], test_orient[2])

        return self.calc_loss(ref_location, ref_img, test_orient)

    def solve_phase1(
        self,
        ref_img: np.ndarray,
        ref_location: tuple[float, float, float],
        run_config: MealRunConfig,
    ):
        obj_func = lambda test_orient: self.calc_invariant_loss(ref_location, ref_img, test_orient)

        bounds = mealpy.FloatVar(lb=OrientUtils.LOWER_BOUND[:2], ub=OrientUtils.UPPER_BOUND[:2])

        problem = mealpy.Problem(
            obj_func=obj_func,
            bounds=bounds,
            minmax="min",
            log_to=None if run_config.silent else "console",
            name="orient_detection_position",
            save_population=run_config.save_pop,
        )

        termination = MealTermination(run_config.max_epoch, run_config.max_time)

        best: Agent = self.optimizer.solve(
            problem=problem,
            mode=run_config.run_mode,
            n_workers=run_config.n_workers,
            termination=termination,
            starting_solutions=None,
            seed=run_config.seed,
        )

        run_hist = RunHistory(self.history.list_epoch_time, self.history.list_global_best_fit)
        pred_position = ObjectPosition(tuple(0, *best.solution), ref_location)
        return pred_position, run_hist

    def solve_phase2(
        self,
        ref_img: np.ndarray,
        ref_location: tuple[float, float, float],
        phase1_solution: tuple[float, float],
        run_config: MealRunConfig,
    ):
        obj_func = lambda test_rotation: self.calc_loss(
            ref_location, ref_img, (test_rotation,) + phase1_solution
        )

        bounds = mealpy.FloatVar(lb=OrientUtils.LOWER_BOUND[:1], ub=OrientUtils.UPPER_BOUND[:1])

        problem = mealpy.Problem(
            obj_func=obj_func,
            bounds=bounds,
            minmax="min",
            log_to=None if run_config.silent else "console",
            name="orient_detection_rotation",
            save_population=run_config.save_pop,
        )

        termination = MealTermination(run_config.max_epoch, run_config.max_time)

        best: Agent = self.optimizer.solve(
            problem=problem,
            mode=run_config.run_mode,
            n_workers=run_config.n_workers,
            termination=termination,
            starting_solutions=None,
            seed=run_config.seed,
        )

        run_hist = RunHistory(self.history.list_epoch_time, self.history.list_global_best_fit)
        pred_position = ObjectPosition((best.solution,) + phase1_solution, ref_location)
        return pred_position, run_hist

    def solve(
        self,
        ref_img: np.ndarray,
        ref_location: tuple[float, float, float],
        run_config: MealRunConfig,
    ) -> tuple[ObjectPosition, RunHistory]:

        solution1, hist = self.solve_phase1(ref_img, ref_location, run_config)

        self.solve_phase2(ref_img, ref_location, solution1, run_config)


class MealAlgorithmPhased(Algorithm):
    def __init__(self, test_viewer: ViewSampler, loss_func: LossFunc, optimizer: mealpy.Optimizer):
        super().__init__(test_viewer, loss_func)
        self.optimizer = optimizer

    @property
    def history(self) -> History:
        return self.optimizer.history

    def get_name(self) -> str:
        return self.optimizer.get_name()

    def get_params(self) -> dict:
        return self.optimizer.get_parameters()

    def solve(
        self,
        ref_img: np.ndarray,
        ref_location: tuple[float, float, float],
        run_config: MealRunConfig,
    ) -> tuple[ObjectPosition, RunHistory]:

        obj_func = lambda test_orient: self.calc_loss(ref_location, ref_img, test_orient)

        bounds = mealpy.FloatVar(lb=OrientUtils.LOWER_BOUND, ub=OrientUtils.UPPER_BOUND)

        problem = mealpy.Problem(
            obj_func=obj_func,
            bounds=bounds,
            minmax="min",
            log_to=None if run_config.silent else "console",
            name="orient_detection",
            save_population=run_config.save_pop,
        )

        termination = MealTermination(run_config.max_epoch, run_config.max_time)

        best: Agent = self.optimizer.solve(
            problem=problem,
            mode=run_config.run_mode,
            n_workers=run_config.n_workers,
            termination=termination,
            starting_solutions=None,
            seed=run_config.seed,
        )

        run_hist = RunHistory(self.history.list_epoch_time, self.history.list_global_best_fit)
        pred_position = ObjectPosition(best.solution, ref_location)
        return pred_position, run_hist
