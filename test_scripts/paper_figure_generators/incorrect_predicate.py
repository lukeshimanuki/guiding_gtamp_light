from gtamp_utils import utils
from gtamp_problem_environments.mover_env import PaPMoverEnv
import numpy as np

problem_env = PaPMoverEnv(problem_idx=0)
obj_poses = [np.array([[3.62234569, -6.92021608, 4.76016588]]), np.array([[1.33077443, -6.0513854, 3.39668951]]),
             np.array([[2.9745574, -6.93927383, 6.25967262]]), np.array([[2.28259212, -6.74139659, 0.51756475]]),
             np.array([[3.32978107, -7.50510825, 3.95933251]]), np.array([[0.06520135, -7.55417455, 4.59608232]]),
             np.array([[1.04589691, -5.68810514, 4.99251205]]), np.array([[0.28712927, -6.10280576, 0.63213569]])]
[utils.set_obj_xytheta(obj_pose, obj) for obj, obj_pose in zip(problem_env.objects, obj_poses)]
kappa = [np.array([1.46718783, -6.71732581, 1.42624871])]
problem_env.env.SetViewer('qtcoin')

viewer = problem_env.env.GetViewer()
camera_transform = np.array([[-0.99993941, -0.00593238, 0.00927237, 1.52711797],
                             [-0.00966105, 0.87669211, -0.48095489, -4.17798805],
                             [-0.00527581, -0.48101533, -0.87669631, 5.99915123],
                             [0., 0., 0., 1.]])
viewer.SetCamera(camera_transform)
utils.set_color('rectangular_packing_box1', [0.7, 0, 0])
utils.visualize_path(kappa)
import pdb;

pdb.set_trace()
