from algs import *
from manipulated_object import ObjectPosition
import loss_funcs
# import mealpy

uni_orients = [[ 0.06391369,  0.12604662,  0.62187402],
       [ 0.4151444 , -1.04497172, -0.87917655],
       [-0.04795201,  0.10837436,  0.27279596],
       [ 1.80137962,  0.97411222, -1.72845657],
       [ 1.72888035,  0.65635599,  1.24976202],
       [ 0.0890371 ,  0.00533704, -0.16308311],
       [ 0.42215688,  0.36192199,  0.54563747],
       [-0.27267751, -0.19914269, -0.68876737],
       [-0.30034492, -0.73378504, -0.36727457],
       [-0.75465701, -0.91582453,  0.09080581],
       [-1.27616636, -0.01714469,  0.38668262],
       [-0.13833163,  0.05613834,  0.5324142 ],
       [ 0.53832904, -1.16726947, -0.36101368],
       [-0.03250202,  0.02793212,  0.0108574 ],
       [ 2.96904739, -0.23666068,  1.15553598],
       [-3.10256847,  0.21701548,  1.5248829 ],
       [-1.79336589, -1.32961435,  1.38045425],
       [-2.67400153,  0.36513492,  1.19786224],
       [ 0.94947097,  0.54207453,  0.4006913 ],
       [ 1.18953515,  0.13478557, -1.54422532]]

uni_positions = [ObjectPosition(orient, (0, 1.3, 0.3)) for orient in uni_orients]


def evaluate(alg:MealAlgorithm, alg_config:MealAlgorithm.Config, loss_func:loss_funcs.LossFunc, path:str):
    import numpy as np
    from view_sampler import ViewSampler, CameraConfig
    
    
    from evaluate import eval_funcs
    

#     from utils.orient import OrientUtils
    from evaluate.evaluator import Evaluator
    # from utils.visualize import SearchPlotter
#     from utils.image import ImageUtils
#     import cv2 as cv
    

    # Create a camera configuration
    cam_config = CameraConfig(location=(0, 0, 0.3), rotation=(np.pi / 2, 0, 0), fov=60)
    world_viewer = ViewSampler("data/hammer/world.xml", cam_config, simulation_time=0)
    sim_viewer = ViewSampler("data/hammer/world_sim.xml", cam_config)

    eval_func=eval_funcs.XorDiff(0.1)

    evaluator = Evaluator(world_viewer, eval_func=eval_func)

    # init_location = (0, 1.3, 0.3)
    # random_orientations = OrientUtils.generate_uniform(3)
    # eval_positions = [ObjectPosition(orient, init_location) for orient in random_orientations]
    
    algorithm = MealAlgorithm(sim_viewer, loss_func, alg())
    # "Ealuations/UniformDet/Mug/IOU/"
    evaluator.enable_logging(path)
    eval_losses = evaluator.evaluate(algorithm, alg_config, uni_positions)
    print(f"{algorithm.get_name()}: {eval_losses}")

    world_viewer.close()
    sim_viewer.close()














