import loss_funcs
from algs import *

import mealpy
from multiprocessing import Process
import evaluation_process as ev



loss_func = loss_funcs.IOU()
alg_config = MealAlgorithm.Config(time_limit=15, silent=True)
root_path = "Evals/OverNight/UniformRand/Hammer/IOU/test"

if __name__ == '__main__':
    for idx, (alg_name, meal_alg) in enumerate(mealpy.get_all_optimizers().items()):
        if "DevSARO" not in alg_name:
            continue
        print(f"===================== Epoch {idx} =====================")
        print(f"Starting Evaluation of: {alg_name}")
        p = Process(target=ev.evaluate, args=(meal_alg,alg_config,loss_func,root_path))
        p.start()
        p.join()



