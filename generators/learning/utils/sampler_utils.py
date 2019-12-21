from generators.learning.utils import data_processing_utils
from gtamp_utils import utils

import pickle
import numpy as np
import time
from matplotlib import pyplot as plt


def prepare_input(smpler_state):
    action = {'pick_abs_base_pose': np.array([0, 0, 0])}
    poses = data_processing_utils.get_processed_poses_from_state(smpler_state, action)[None, :]

    goal_flags = smpler_state.goal_flags
    collisions = smpler_state.collision_vector

    poses = poses[:, :20]

    return goal_flags, collisions, poses


def generate_smpl_batch(concrete_state, sampler, noise_batch, key_configs):
    goal_flags, collisions, poses = prepare_input(concrete_state)

    # processing key configs
    # todo below can be saved for this state as well
    stime = time.time()
    xmin = -0.7
    xmax = 4.3
    ymin = -8.55
    ymax = -4.85
    indices_to_delete = np.hstack([np.where(key_configs[:, 1] > ymax)[0], np.where(key_configs[:, 1] < ymin)[0],
                                   np.where(key_configs[:, 0] > xmax)[0], np.where(key_configs[:, 0] < xmin)[0]])
    key_configs = np.delete(key_configs, indices_to_delete, axis=0)
    collisions = np.delete(collisions, indices_to_delete, axis=1)
    goal_flags = np.delete(goal_flags, indices_to_delete, axis=1)
    #print "delete time:", time.time() - stime

    # todo these following three lines can be removed
    stime = time.time()
    key_configs = np.array([utils.encode_pose_with_sin_and_cos_angle(p) for p in key_configs])
    key_configs = key_configs.reshape((1, len(key_configs), 4, 1))
    key_configs = key_configs.repeat(len(poses), axis=0)
    #print "key config processing time:", time.time() - stime

    # make repeated inputs other than noise, because we are making multiple predictions
    # todo save the following to the concrete state
    stime = time.time()
    n_smpls = len(noise_batch)
    goal_flags = np.tile(goal_flags, (n_smpls, 1, 1, 1))
    key_configs = np.tile(key_configs, (n_smpls, 1, 1, 1))
    collisions = np.tile(collisions, (n_smpls, 1, 1, 1))
    poses = np.tile(poses, (n_smpls, 1))
    if len(noise_batch) > 1:
        noise_batch = np.array(noise_batch).squeeze()
    #print "tiling time:", time.time() - stime

    inp = [goal_flags, key_configs, collisions, poses, noise_batch]
    stime = time.time()
    pred_batch = sampler.policy_model.predict(inp)
    #print "prediction time:", time.time() - stime
    stime = time.time()
    samples_in_se2 = [utils.decode_pose_with_sin_and_cos_angle(q) for q in pred_batch]
    #print "Decoding time: ", time.time() - stime
    return samples_in_se2


def generate_policy_smpl_batch(smpler_state, policy, noise_batch):
    goal_flags, collisions, poses = prepare_input(smpler_state)
    obj = smpler_state.obj
    obj_pose = utils.clean_pose_data(smpler_state.abs_obj_pose)

    smpler_state.abs_obj_pose = obj_pose
    goal_flags = smpler_state.goal_flags
    collisions = smpler_state.collision_vector

    key_configs = pickle.load(open('prm.pkl', 'r'))[0]
    key_configs = np.delete(key_configs, [415, 586, 615, 618, 619], axis=0)

    xmin = -0.7;
    xmax = 4.3
    ymin = -8.55;
    ymax = -4.85
    indices_to_delete = np.hstack([np.where(key_configs[:, 1] > ymax)[0], np.where(key_configs[:, 1] < ymin)[0],
                                   np.where(key_configs[:, 0] > xmax)[0], np.where(key_configs[:, 0] < xmin)[0]])
    key_configs = np.delete(key_configs, indices_to_delete, axis=0)
    collisions = np.delete(collisions, indices_to_delete, axis=1)
    goal_flags = np.delete(goal_flags, indices_to_delete, axis=1)

    key_configs = np.array([utils.encode_pose_with_sin_and_cos_angle(p) for p in key_configs])
    key_configs = key_configs.reshape((1, len(key_configs), 4, 1))
    key_configs = key_configs.repeat(len(poses), axis=0)

    n_smpls = len(noise_batch)
    goal_flags = np.tile(goal_flags, (n_smpls, 1, 1, 1))
    key_configs = np.tile(key_configs, (n_smpls, 1, 1, 1))
    collisions = np.tile(collisions, (n_smpls, 1, 1, 1))
    poses = np.tile(poses, (n_smpls, 1))
    if len(noise_batch) > 1:
        noise_batch = np.array(noise_batch).squeeze()

    inp = [goal_flags, key_configs, collisions, poses, noise_batch]
    pred_batch = policy.policy_model.predict(inp)
    if pred_batch.shape[-1] == 8:
        picks = []
        places = []
        for q in pred_batch:
            pick = utils.decode_pose_with_sin_and_cos_angle(q[0:4])
            place = utils.decode_pose_with_sin_and_cos_angle(q[4:])
            picks.append(pick)
            places.append(place)
        import pdb;
        pdb.set_trace()
        utils.visualize_placements(places, obj)
        utils.visualize_path(picks[0:50])
    else:
        decoded = [utils.decode_pose_with_sin_and_cos_angle(q) for q in pred_batch]
        utils.visualize_placements(decoded, obj)

    # value_net = policy.value_model.predict([poses, key_configs, collisions, goal_flags]).squeeze()
    # eval_net = policy.evalnet_model.predict([poses, key_configs, collisions, goal_flags]).squeeze()
    # value_net = [utils.decode_pose_with_sin_and_cos_angle(p) for p in value_net]
    # best_qk = policy.best_qk_model.predict([goal_flags, key_configs, collisions, poses]).squeeze()
    """
    eval_net = policy.evalnet_model.predict([goal_flags, key_configs, collisions, poses]).squeeze()
    idxs = eval_net[0] > 1e-5
    key_configs = key_configs[0].squeeze()
    konfs = np.array([utils.decode_pose_with_sin_and_cos_angle(k) for k in key_configs])
    import pdb;pdb.set_trace()
    """

    import pdb;
    pdb.set_trace()

    # H, xedges, yedges = np.histogram2d(pred_batch[:, 0], pred_batch[:, 1], bins=10)
    # plt.imshow(H, interpolation='nearest', origin='low', extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]])
    # utils.visualize_path([best_qk])
    # x = np.array([pred_batch[0,0],pred_batch[0,1], pred_batch[0,2], pred_batch[0,3]]) + 0.5
    # return np.vstack([pred_batch,x])
    # return place_smpl
