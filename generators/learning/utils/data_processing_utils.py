from gtamp_utils import utils
import numpy as np

state_data_mode = 'absolute'
action_data_mode = 'PICK_grasp_params_and_abs_base_PLACE_abs_base'
action_data_mode = 'PICK_grasp_params_and_ir_parameters_PLACE_abs_base'


def make_konfs_relative_to_pose(obj_pose, key_configs):
    rel_konfs = []
    utils.clean_pose_data(obj_pose)
    for k in key_configs:
        konf = utils.clean_pose_data(k)
        rel_konf = utils.get_relative_robot_pose_wrt_body_pose(konf, obj_pose)
        rel_konf = utils.encode_pose_with_sin_and_cos_angle(rel_konf)
        rel_konfs.append(rel_konf)
    return np.array(rel_konfs)


def get_absolute_placement_from_relative_placement(rel_placement, obj_abs_pose):
    rel_placement = utils.decode_pose_with_sin_and_cos_angle(rel_placement)
    if action_data_mode == 'pick_parameters_place_relative_to_object':
        # abs_place = placement.squeeze() + obj_abs_pose.squeeze()
        abs_place = utils.get_absolute_pose_from_relative_pose(rel_placement, obj_abs_pose)
    else:
        raise NotImplementedError

    return abs_place


def get_processed_poses_from_state(state, action):
    if state_data_mode == 'absolute':
        obj_pose = utils.encode_pose_with_sin_and_cos_angle(state.abs_obj_pose)
        curr_robot_pose = utils.encode_pose_with_sin_and_cos_angle(state.abs_robot_pose)
        goal_obj_poses = np.hstack([utils.encode_pose_with_sin_and_cos_angle(o) for o in state.abs_goal_obj_poses])
    elif state_data_mode == 'robot_rel_to_obj':
        obj_pose = utils.encode_pose_with_sin_and_cos_angle(state.abs_obj_pose)
        # this is the initial robot pose, before picking an object. Is the collision information while holding the obj?
        robot_pose = utils.get_relative_robot_pose_wrt_body_pose(state.abs_robot_pose, state.abs_obj_pose)
        curr_robot_pose = utils.encode_pose_with_sin_and_cos_angle(robot_pose)
        # I must preserve the locations different objects
        goal_obj_poses = [utils.get_relative_robot_pose_wrt_body_pose(o, state.abs_obj_pose) for o in
                          state.abs_goal_obj_poses]
        goal_obj_poses = [utils.encode_pose_with_sin_and_cos_angle(o) for o in goal_obj_poses]
        goal_obj_poses = np.hstack(goal_obj_poses)
    else:
        raise not NotImplementedError
    pose = np.hstack([obj_pose, goal_obj_poses, curr_robot_pose])
    return pose


def put_pose_wrt_region(pose, region):
    if region == 'home_region':
        region_box = [[-1.75, -3.16], [5.25, 3.16]]
    elif region == 'loading_region':
        region_box = [[-0.7, -8.55], [4.3, -4.85]]
    else:
        raise NotImplementedError

    pose[0] = pose[0] - region_box[0][0]
    pose[1] = pose[1] - region_box[0][1]
    return pose


def convert_pose_rel_to_region_to_abs_pose(pose, region):
    if region == 'home_region':
        region_box = [[-1.75, -3.16], [5.25, 3.16]]
    elif region == 'loading_region':
        region_box = [[-0.7, -8.55], [4.3, -4.85]]
    else:
        raise NotImplementedError

    pose[0] = pose[0] + region_box[0][0]
    pose[1] = pose[1] + region_box[0][1]
    return pose


def unnormalize_pose_wrt_region(pose, region):
    if region == 'home_region':
        region_box = [[-1.75, -3.16], [5.25, 3.16]]
    elif region == 'loading_region':
        region_box = [[-0.7, -8.55], [4.3, -4.85]]
    else:
        raise NotImplementedError
    size_x, size_y = get_box_size(region_box)
    pose[0] = pose[0] * size_x
    pose[1] = pose[1] * size_y
    pose = convert_pose_rel_to_region_to_abs_pose(pose, region)
    return pose


def get_place_pose_wrt_region(pose, region):
    place_pose = put_pose_wrt_region(pose, region)
    return place_pose


def get_box_size(box):
    box_size_x = box[1][0] - box[0][0]
    box_size_y = box[1][1] - box[0][1]
    return box_size_x, box_size_y


def normalize_place_pose_wrt_region(pose, region):
    if region == 'home_region':
        region_box = [[-1.75, -3.16], [5.25, 3.16]]
    elif region == 'loading_region':
        region_box = [[-0.7, -8.55], [4.3, -4.85]]
    else:
        raise NotImplementedError
    place_pose = put_pose_wrt_region(pose, region)
    size_x, size_y = get_box_size(region_box)
    place_pose[0] = place_pose[0] / size_x
    place_pose[1] = place_pose[1] / size_y
    return place_pose


def is_q_close_enough_to_any_config_in_motion(q, motion, xy_threshold=0.1):
    q = np.array(q)
    for c in motion:
        c = c.squeeze()
        xy_dist = np.linalg.norm(c[0:2] - q[0:2])
        if xy_dist < xy_threshold:
            return True
    return False


def get_relevance_info(konf, collisions, motion):
    labels = []
    for key_config, collision in zip(konf, collisions):
        label = not collision and is_q_close_enough_to_any_config_in_motion(key_config, motion)
        labels.append(label)
    no_collision_at_relevant_konf = collisions[labels].sum() == 0
    assert no_collision_at_relevant_konf
    return np.array(labels)


def get_processed_poses_from_action(state, action):
    # grasp_params abs_pick_pose
    # grasp params ir_parameters
    # 'PICK_grasp_params_and_abs_base_PLACE_abs_base'

    if 'PICK_grasp_params_and_abs_base' in action_data_mode:
        grasp_params = action['pick_base_ir_parameters'][0:3][None, :]
        abs_pick_pose = utils.encode_pose_with_sin_and_cos_angle(action['pick_abs_base_pose'])[None, :]
        pick_params = np.hstack([grasp_params, abs_pick_pose])[None, :]
    elif 'PICK_grasp_params_and_ir_parameters' in action_data_mode:
        abs_pick_pose = action['pick_abs_base_pose']
        portion, base_angle, facing_angle_offset \
            = utils.get_ir_parameters_from_robot_obj_poses(abs_pick_pose, state.abs_obj_pose)
        base_angle = utils.encode_angle_in_sin_and_cos(base_angle)
        grasp_params = action['pick_base_ir_parameters'][0:3]
        pick_params = np.hstack([grasp_params, portion, base_angle, facing_angle_offset])[None, :]
    else:
        raise NotImplementedError

    if 'PLACE_abs_base' in action_data_mode:
        place_params = utils.encode_pose_with_sin_and_cos_angle(action['place_obj_abs_pose'])[None, :]
    else:
        raise NotImplementedError

    action = np.hstack([pick_params, place_params])
    return action
