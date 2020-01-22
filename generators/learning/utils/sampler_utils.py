from generators.learning.utils import data_processing_utils
from generators.learning.PlacePolicyIMLE import gaussian_noise
from gtamp_utils import utils

import pickle
import numpy as np
import time


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


def get_indices_to_delete(region, key_configs):
    if region == 'loading_region':
        xmin = -0.7
        xmax = 4.3
        ymin = -8.55
        ymax = -4.85
    elif region == 'home_region':
        xmin = -1.11322709
        xmax = 4.99456405
        ymin = -2.9463328
        ymax = 2.54926346
    indices_to_delete = np.hstack([np.where(key_configs[:, 1] > ymax)[0], np.where(key_configs[:, 1] < ymin)[0],
                                   np.where(key_configs[:, 0] > xmax)[0], np.where(key_configs[:, 0] < xmin)[0]])
    return indices_to_delete


def prepare_input_except_noise(smpler_state, delete=False, region=None, filter_konfs=False):
    poses = data_processing_utils.get_processed_poses_from_state(smpler_state, None)[None, :]
    obj_pose = utils.clean_pose_data(smpler_state.abs_obj_pose)

    smpler_state.abs_obj_pose = obj_pose
    goal_flags = smpler_state.goal_flags
    collisions = smpler_state.pick_collision_vector

    key_configs = pickle.load(open('prm.pkl', 'r'))[0]
    key_configs = np.delete(key_configs, [415, 586, 615, 618, 619], axis=0)
    if delete:
        indices_to_delete = get_indices_to_delete(region, key_configs)
        key_configs = np.delete(key_configs, indices_to_delete, axis=0)
        collisions = np.delete(collisions, indices_to_delete, axis=1)
        goal_flags = np.delete(goal_flags, indices_to_delete, axis=1)

    if filter_konfs:
        key_configs = data_processing_utils.filter_configs_that_are_too_close(key_configs)

    key_configs = np.array([utils.encode_pose_with_sin_and_cos_angle(p) for p in key_configs])
    key_configs = key_configs.reshape((1, len(key_configs), 4, 1))
    key_configs = key_configs.repeat(len(poses), axis=0)
    inp = [goal_flags, key_configs, collisions, poses]
    inp = {'goal_flags': goal_flags, 'key_configs': key_configs, 'collisions': collisions, 'poses': poses}
    return inp


def prepare_input(smpler_state, noise_batch, delete=False, region=None, filter_konfs=False):
    poses = data_processing_utils.get_processed_poses_from_state(smpler_state, None)[None, :]
    obj_pose = utils.clean_pose_data(smpler_state.abs_obj_pose)

    smpler_state.abs_obj_pose = obj_pose
    goal_flags = smpler_state.goal_flags
    collisions = smpler_state.pick_collision_vector

    key_configs = pickle.load(open('prm.pkl', 'r'))[0]
    key_configs = np.delete(key_configs, [415, 586, 615, 618, 619], axis=0)
    if delete:
        indices_to_delete = get_indices_to_delete(region, key_configs)
        key_configs = np.delete(key_configs, indices_to_delete, axis=0)
        collisions = np.delete(collisions, indices_to_delete, axis=1)
        goal_flags = np.delete(goal_flags, indices_to_delete, axis=1)

    if filter_konfs:
        key_configs = data_processing_utils.filter_configs_that_are_too_close(key_configs)

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


def make_predictions(smpler_state, smpler, noise_batch):
    inp = prepare_input(smpler_state, noise_batch, delete=True, region='loading_region')
    pred_batch = smpler.policy_model.predict(inp)
    return pred_batch


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
    very_stime = time.time()
    pick_smpler = policy['pick']
    stime = time.time()
    inp = prepare_input(smpler_state, noise_batch, delete=True, region='loading_region', filter_konfs=False)
    print "preparation_time_1", time.time() - stime

    pick_samples = pick_smpler.policy_model.predict(inp)

    # preparation for place sampler
    pick_base_poses = []
    stime = time.time()

    # todo generate place for picks that are actually going to be used. This would almost half the prediction time.
    #   In this scheme, what happens if we generate like 1 sample at a time?
    for p in pick_samples:
        ir_parameters = p[3:]
        portion = ir_parameters[0]
        base_angle = utils.decode_sin_and_cos_to_angle(ir_parameters[1:3])
        facing_angle_offset = ir_parameters[3]
        pick_param = np.hstack([p[:3], portion, base_angle, facing_angle_offset])
        _, pick_base_pose = utils.get_pick_base_pose_and_grasp_from_pick_parameters(smpler_state.obj, pick_param)
        # pick_params.append({'q_goal': pick_base_pose})
        pick_base_pose = utils.encode_pose_with_sin_and_cos_angle(pick_base_pose)
        pick_base_poses.append(pick_base_pose)
    pick_base_poses = np.array(pick_base_poses)
    print "pick processing time", time.time() - stime

    # making place samples based on pick base poses
    stime = time.time()
    inp = prepare_input(smpler_state, noise_batch, delete=True, region=smpler_state.region, filter_konfs=False)
    print "preparation_time_2", time.time() - stime

    poses = inp[-2]
    poses[:, -4:] = pick_base_poses
    inp[-2] = poses
    z_smpls = gaussian_noise(z_size=(len(noise_batch), 4))
    inp[-1] = z_smpls
    place_smpler = policy['place']

    stime = time.time()
    place_samples = place_smpler.policy_model.predict(inp)
    print "place_prediction_time", time.time() - stime
    # place_sample_values = place_smpler.value_model.predict(inp)
    # place_sample_values = [utils.decode_pose_with_sin_and_cos_angle(p) for p in place_sample_values[0]]
    print "Total time taken", time.time() - very_stime

    return pick_samples, place_samples
