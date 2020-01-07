from generators.learning.utils import data_processing_utils
from gtamp_utils import utils

import pickle
import numpy as np
import time


def gaussian_noise(z_size):
    return np.random.normal(size=z_size, scale=0.5).astype('float32')


def unprocess_pick_and_place_smpls(smpls):
    pick_smpls = smpls[0]
    place_smpls = smpls[1]

    pick_unprocessed = []
    place_unprocessed = []
    for pick_smpl, place_smpl in zip(pick_smpls, place_smpls):
        grasp_params = pick_smpl[0:3]
        ir_parameters = pick_smpl[3:]
        portion = ir_parameters[0]
        base_angle = utils.decode_sin_and_cos_to_angle(ir_parameters[1:3])
        facing_angle_offset = ir_parameters[3]
        pick_unprocessed.append(np.hstack([grasp_params, portion, base_angle, facing_angle_offset]))

        abs_base_pose = utils.decode_pose_with_sin_and_cos_angle(place_smpl)
        place_unprocessed.append(abs_base_pose)
    smpls = np.hstack([pick_unprocessed, place_unprocessed])
    return smpls


def prepare_input(smpler_state, noise_batch):
    poses = data_processing_utils.get_processed_poses_from_state(smpler_state, None)[None, :]
    obj_pose = utils.clean_pose_data(smpler_state.abs_obj_pose)

    smpler_state.abs_obj_pose = obj_pose
    goal_flags = smpler_state.goal_flags
    collisions = smpler_state.pick_collision_vector

    key_configs = pickle.load(open('prm.pkl', 'r'))[0]
    key_configs = np.delete(key_configs, [415, 586, 615, 618, 619], axis=0)
    xmin = -0.7
    xmax = 4.3
    ymin = -8.55
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
    return inp


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
    # print "delete time:", time.time() - stime

    # todo these following three lines can be removed
    stime = time.time()
    key_configs = np.array([utils.encode_pose_with_sin_and_cos_angle(p) for p in key_configs])
    key_configs = key_configs.reshape((1, len(key_configs), 4, 1))
    key_configs = key_configs.repeat(len(poses), axis=0)
    # print "key config processing time:", time.time() - stime

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
    # print "tiling time:", time.time() - stime

    inp = [goal_flags, key_configs, collisions, poses, noise_batch]
    stime = time.time()
    pred_batch = sampler.policy_model.predict(inp)
    # print "prediction time:", time.time() - stime
    stime = time.time()
    samples_in_se2 = [utils.decode_pose_with_sin_and_cos_angle(q) for q in pred_batch]
    # print "Decoding time: ", time.time() - stime
    return samples_in_se2


def make_predictions(smpler_state, smpler, noise_batch):
    inp = prepare_input(smpler_state, noise_batch)
    pred_batch = smpler.policy_model.predict(inp)
    return pred_batch


def generate_pick_or_place_batch(smpler_state, policy, noise_batch):
    pred_batch = make_predictions(smpler_state, policy, noise_batch)
    samples_in_se2 = np.array([utils.decode_pose_with_sin_and_cos_angle(q) for q in pred_batch])
    return samples_in_se2


def get_konf_obstacles_while_holding(pick_samples, sampler_state, problem_env):
    konf_obstacles_while_holding = []
    xmin = -0.7
    xmax = 4.3
    ymin = -8.55
    ymax = -4.85
    key_configs = pickle.load(open('prm.pkl', 'r'))[0]
    key_configs = np.delete(key_configs, [415, 586, 615, 618, 619], axis=0)
    indices_to_delete = np.hstack([np.where(key_configs[:, 1] > ymax)[0], np.where(key_configs[:, 1] < ymin)[0],
                                   np.where(key_configs[:, 0] > xmax)[0], np.where(key_configs[:, 0] < xmin)[0]])
    sampler_state.key_configs = np.delete(key_configs, indices_to_delete, axis=0)
    for pick_smpl in pick_samples:
        utils.two_arm_pick_object(sampler_state.obj, pick_smpl)
        sampler_state.place_collision_vector = sampler_state.get_collison_vector(None)
        konf_obstacles_while_holding.append(sampler_state.place_collision_vector)
        utils.two_arm_place_object(pick_smpl)
    return np.array(konf_obstacles_while_holding).reshape((len(pick_samples), 291, 2, 1))


def generate_pick_and_place_batch(smpler_state, policy, noise_batch):
    pick_smpler = policy['pick']
    inp = prepare_input(smpler_state, noise_batch)
    pick_samples = pick_smpler.policy_model.predict(inp)

    # preparation for place sampler
    pick_base_poses = []
    pick_params = []
    for p in pick_samples:
        ir_parameters = p[3:]
        portion = ir_parameters[0]
        base_angle = utils.decode_sin_and_cos_to_angle(ir_parameters[1:3])
        facing_angle_offset = ir_parameters[3]
        pick_param = np.hstack([p[:3], portion, base_angle, facing_angle_offset])
        _, pick_base_pose = utils.get_pick_base_pose_and_grasp_from_pick_parameters(smpler_state.obj, pick_param)
        pick_params.append({'q_goal': pick_base_pose})
        pick_base_pose = utils.encode_pose_with_sin_and_cos_angle(pick_base_pose)
        pick_base_poses.append(pick_base_pose)
    pick_base_poses = np.array(pick_base_poses)

    # todo I need to make a separate key config obstacles for place sampler
    # place_konf_obstacles = get_konf_obstacles_while_holding(pick_params, smpler_state, problem_env)
    # inp[2] = place_konf_obstacles
    poses = inp[-2]
    poses[:, -4:] = pick_base_poses
    inp[-2] = poses
    z_smpls = gaussian_noise(z_size=(len(noise_batch), 4))
    inp[-1] = z_smpls
    place_smpler = policy['place']
    place_samples = place_smpler.policy_model.predict(inp)
    """
    picks = []
    places = []
    pred_batch = make_predictions(smpler_state, policy, noise_batch)
    for q in pred_batch:
        pick = utils.decode_pose_with_sin_and_cos_angle(q[0:4])
        place = utils.decode_pose_with_sin_and_cos_angle(q[4:])
        picks.append(pick)
        places.append(place)
    return (picks, places)
    """
    return pick_samples, place_samples
