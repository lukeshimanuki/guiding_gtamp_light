from manipulation.primitives.transforms import *
import manipulation
from manipulation.bodies.bodies import *
from manipulation.primitives.transforms import get_point, set_point, pose_from_quat_point, unit_quat
from manipulation.primitives.utils import mirror_arm_config

import copy
import openravepy
import numpy as np
import math
import time
import scipy as sp
from scipy import spatial as sp_spatial
from openravepy import DOFAffine, RaveCreateKinBody, RaveCreateRobot

FOLDED_LEFT_ARM = [0.0, 1.29023451, 0.0, -2.121308, 0.0, -0.69800004, 0.0]

PR2_ARM_LENGTH = 0.9844


def get_color_of(body):
    env = openravepy.RaveGetEnvironments()[0]
    if type(body) == unicode or type(body) == str:
        obj = env.GetKinBody(body)
    else:
        obj = body

    return get_color(obj)


def convert_binary_vec_to_one_hot(collision_vector):
    n_konf = collision_vector.shape[0]
    one_hot_cvec = np.zeros((n_konf, 2))
    one_hot_cvec[:, 0] = collision_vector
    one_hot_cvec[:, 1] = 1 - collision_vector
    return one_hot_cvec


def compute_angle_to_be_set(target_xy, src_xy):
    target_dirn = target_xy - src_xy
    target_dirn = target_dirn / np.linalg.norm(target_dirn)
    if target_dirn[1] < 0:
        # rotation from x-axis, because that is the default rotation
        angle_to_be_set = -math.acos(np.dot(target_dirn, np.array(([1, 0]))))
    else:
        angle_to_be_set = math.acos(np.dot(target_dirn, np.array(([1, 0]))))
    return angle_to_be_set


def convert_rel_to_abs_base_pose(rel_xytheta, src_xy):
    if len(rel_xytheta.shape) == 1: rel_xytheta = rel_xytheta[None, :]
    assert (len(src_xy.shape) == 1)
    ndata = rel_xytheta.shape[0]
    abs_base_pose = np.zeros((ndata, 3))
    abs_base_pose[:, 0:2] = rel_xytheta[:, 0:2] + src_xy[0:2]
    for i in range(ndata):
        th_to_be_set = compute_angle_to_be_set(src_xy[0:2], abs_base_pose[i, 0:2])
        abs_base_pose[i, -1] = th_to_be_set + rel_xytheta[i, -1]
    return abs_base_pose


def set_body_transparency(body, transparency):
    env = openravepy.RaveGetEnvironments()[0]
    if type(body) == unicode or type(body) == str:
        body = env.GetKinBody(body)
    for link in body.GetLinks():
        for geom in link.GetGeometries():
            geom.SetTransparency(transparency)


def set_obj_xytheta(xytheta, obj):
    if isinstance(xytheta, list) or isinstance(xytheta, tuple):
        xytheta = np.array(xytheta)
    env = openravepy.RaveGetEnvironments()[0]
    if type(obj) == unicode or type(obj) == str:
        obj = env.GetKinBody(obj)

    xytheta = xytheta.squeeze()
    set_quat(obj, quat_from_angle_vector(xytheta[-1], np.array([0, 0, 1])))
    set_xy(obj, xytheta[0], xytheta[1])


def set_active_config(conf, robot=None):
    if robot is None:
        env = openravepy.RaveGetEnvironments()[0]
        robot = env.GetRobot('pr2')
    robot.SetActiveDOFValues(conf.squeeze())


def draw_obj_at_conf(conf, transparency, name, obj, env, color=None):
    before = get_body_xytheta(obj)
    newobj = RaveCreateKinBody(env, '')
    newobj.Clone(obj, 0)
    newobj.SetName(name)
    env.Add(newobj, True)
    set_obj_xytheta(conf, newobj)
    newobj.Enable(False)
    if color is not None:
        set_color(newobj, color)
    set_body_transparency(newobj, transparency)


def draw_robot_at_conf(conf, transparency, name, robot, env, color=None):
    held_obj = robot.GetGrabbed()
    before = get_body_xytheta(robot)
    newrobot = RaveCreateRobot(env, '')
    newrobot.Clone(robot, 0)
    newrobot.SetName(name)
    env.Add(newrobot, True)
    set_active_config(conf, newrobot)
    newrobot.Enable(False)
    if color is not None:
        set_color(newrobot, color)

    if len(held_obj) > 0:
        print get_body_xytheta(held_obj[0])
        set_robot_config(conf)
        held_obj = robot.GetGrabbed()[0]
        held_obj_trans = held_obj.GetTransform()
        release_obj()
        new_obj = RaveCreateKinBody(env, '')
        new_obj.Clone(held_obj, 0)
        new_obj.SetName(name + '_obj')
        env.Add(new_obj, True)
        new_obj.SetTransform(held_obj_trans)
        for link in new_obj.GetLinks():
            for geom in link.GetGeometries():
                geom.SetTransparency(transparency)
        grab_obj(held_obj)
        set_robot_config(before)

    for link in newrobot.GetLinks():
        for geom in link.GetGeometries():
            geom.SetTransparency(transparency)


def visualize_pick_and_place(pick, place):
    env = openravepy.RaveGetEnvironments()[0]
    obj = pick.discrete_parameters['object']
    if type(obj) == unicode or type(obj) == str:
        obj = env.GetKinBody(obj)

    saver = CustomStateSaver(env)
    visualize_path(pick.low_level_motion)
    two_arm_pick_object(obj, pick.continuous_parameters)
    visualize_path(place.low_level_motion)
    saver.Restore()


def visualize_placements(placements, obj=None):
    assert len(openravepy.RaveGetEnvironments()) == 1
    env = openravepy.RaveGetEnvironments()[0]
    if obj is None:
        obj = env.GetRobots()[0].GetGrabbed()[0]
    if type(obj) == unicode or type(obj) == str:
        obj = env.GetKinBody(obj)
    for idx, conf in enumerate(placements):
        draw_obj_at_conf(conf, 0.7, 'place' + str(idx), obj, env)
    raw_input("Continue?")
    remove_drawn_configs('place', env)


def visualize_path(path):
    assert len(openravepy.RaveGetEnvironments()) == 1
    env = openravepy.RaveGetEnvironments()[0]
    robot = env.GetRobot('pr2')
    path = [p.squeeze() for p in path]
    dim_path = len(path[0])
    if dim_path == 3:
        robot.SetActiveDOFs([], DOFAffine.X | DOFAffine.Y | DOFAffine.RotationAxis, [0, 0, 1])
    elif dim_path == 11:
        manip = robot.GetManipulator('rightarm_torso')
        robot.SetActiveDOFs(manip.GetArmIndices(), DOFAffine.X | DOFAffine.Y | DOFAffine.RotationAxis, [0, 0, 1])
    else:
        assert path[0].shape[0] == robot.GetActiveDOF(), 'robot and path should have same dof'

    env = robot.GetEnv()
    if len(path) > 1000:
        path_reduced = path[0:len(path) - 1:int(len(path) * 0.1)]
    else:
        path_reduced = path
    for idx, conf in enumerate(path_reduced):
        is_goal_config = idx == len(path_reduced) - 1
        if is_goal_config:
            draw_robot_at_conf(conf, 0.5, 'path' + str(idx), robot, env)
        else:
            draw_robot_at_conf(conf, 0.7, 'path' + str(idx), robot, env)
    raw_input("Continue?")
    remove_drawn_configs('path', env)


def open_gripper(robot=None):
    assert len(openravepy.RaveGetEnvironments()) == 1
    env = openravepy.RaveGetEnvironments()[0]
    robot = env.GetRobot('pr2')
    robot.SetDOFValues(np.array([0.54800022]), robot.GetActiveManipulator().GetGripperIndices())


def close_gripper(robot):
    robot.SetDOFValues(np.array([0]), robot.GetActiveManipulator().GetGripperIndices())
    # taskprob = interfaces.TaskManipulation(robot)
    # robot.GetEnv().StopSimulation()
    # taskprob.CloseFingers()
    # robot.GetEnv().StartSimulation(0.01)


def check_collision_except(exception_body, env):
    # todo make this more efficient
    assert exception_body != env.GetRobots()[0], 'Collision exception cannot be the robot'

    # exception_body.Enable(False)  # todo what happens to the attached body when I enable and disable the held object?
    # col = env.CheckCollision(env.GetRobots()[0])
    # exception_body.Enable(True)
    # todo optimize this later
    return np.any([env.CheckCollision(env.GetRobots()[0], body) for body in env.GetBodies() if body != exception_body])
    # return col


def set_robot_config(base_pose, robot=None):
    if robot is None:
        env = openravepy.RaveGetEnvironments()[0]
        robot = env.GetRobot('pr2')

    base_pose = np.array(base_pose)
    base_pose = clean_pose_data(base_pose.astype('float'))

    robot.SetActiveDOFs([], DOFAffine.X | DOFAffine.Y | DOFAffine.RotationAxis, [0, 0, 1])
    base_pose = np.array(base_pose).squeeze()
    """
    while base_pose[-1] < 0:
      try:
        factor = -int(base_pose[-1] /(2*np.pi))
      except:
        import pdb;pdb.set_trace()
      if factor == 0: factor = 1
      base_pose[-1] += factor*2*np.pi
    while base_pose[-1] > 2*np.pi:
      factor = int(base_pose[-1] /(2*np.pi))
      base_pose[-1] -= factor*2*np.pi
    
    if base_pose[-1] <
    if base_pose[-1] > 1.01:
      base_pose[-1] = 1.01
    elif base_pose[-1] < 0.99:
      base_pose[-1] = 0.99
    """
    # print base_pose
    robot.SetActiveDOFValues(base_pose)


def trans_from_xytheta(obj, xytheta):
    rot = rot_from_quat(quat_from_z_rot(xytheta[-1]))
    z = get_point(obj)[-1]
    trans = np.eye(4)
    trans[:3, :3] = rot
    trans[:3, -1] = [xytheta[0], xytheta[1], z]
    return trans


def remove_drawn_configs(name, env=None):
    if env is None:
        assert len(openravepy.RaveGetEnvironments()) == 1
        env = openravepy.RaveGetEnvironments()[0]

    for body in env.GetBodies():
        if body.GetName().find(name) != -1:
            env.Remove(body)


def draw_robot_base_configs(configs, robot, env, name='bconf', transparency=0.7):
    for i in range(len(configs)):
        config = configs[i]
        draw_robot_at_conf(config, transparency, name + str(i), robot, env)


def draw_configs(configs, env, name='point', colors=None, transparency=0.1):
    # assert configs[0].shape==(6,), 'Config shape must be (6,)'
    if colors is None:
        for i in range(len(configs)):
            config = configs[i]
            new_body = box_body(env, 0.1, 0.05, 0.05, \
                                name=name + '%d' % i, \
                                color=(1, 0, 0), \
                                transparency=transparency)
            env.Add(new_body);
            set_point(new_body, np.append(config[0:2], 0.075))
            new_body.Enable(False)
            th = config[2]
            set_quat(new_body, quat_from_z_rot(th))
    else:
        for i in range(len(configs)):
            config = configs[i]
            if isinstance(colors, tuple):
                color = colors
            else:
                color = colors[i]
            new_body = box_body(env, 0.1, 0.05, 0.05, \
                                name=name + '%d' % i, \
                                color=color, \
                                transparency=transparency)
            """
            new_body = load_body(env,'mug.xml')
            set_name(new_body, name+'%d'%i)
            set_transparency(new_body, transparency)
            """
            env.Add(new_body);
            set_point(new_body, np.append(config[0:2], 0.075))
            new_body.Enable(False)
            th = config[2]
            set_quat(new_body, quat_from_z_rot(th))


def get_trajectory_length(trajectory):
    dists = 0
    for i in range(len(trajectory) - 1):
        dists += se2_distance(trajectory[i + 1], trajectory[i], 1, 1)
    return dists


def clean_pose_data(pose_data):
    if len(pose_data.shape) > 0 and pose_data.size > 0:
        # fixes angle to be between 0 to 2pi
        if len(pose_data.shape) == 1:
            pose_data = pose_data[None, :]

        data_idx_neg_angles = pose_data[:, -1] < 0
        data_idx_big_angles = pose_data[:, -1] > 2 * np.pi
        pose_data[data_idx_neg_angles, -1] += 2 * np.pi
        pose_data[data_idx_big_angles, -1] -= 2 * np.pi

        # assert( np.all(pose_data[:,-1]>=0) and np.all(pose_data[:,-1] <2*np.pi))
        return pose_data
    else:
        return np.array([])[None, :]


def compute_occ_vec(key_configs):
    env = openravepy.RaveGetEnvironments()[0]
    robot = env.GetRobot('pr2')
    occ_vec = []
    with robot:
        for config in key_configs:
            set_robot_config(config, robot)
            collision = env.CheckCollision(robot) * 1
            occ_vec.append(collision)
    return np.array(occ_vec)


def get_robot_xytheta(robot=None):
    if robot is None:
        env = openravepy.RaveGetEnvironments()[0]
        robot = env.GetRobot('pr2')
    with robot:
        robot.SetActiveDOFs([], DOFAffine.X | DOFAffine.Y | DOFAffine.RotationAxis, [0, 0, 1])
        robot_xytheta = robot.GetActiveDOFValues()
    robot_xytheta = robot_xytheta[None, :]
    clean_pose_data(robot_xytheta)
    return robot_xytheta


def get_relative_robot_pose_wrt_body_pose(robot_pose, body_pose):
    t_robot = get_transform_from_pose(robot_pose, 'robot')
    t_obj = get_transform_from_pose(body_pose, 'kinbody')
    rel_t = get_relative_transform_T1_wrt_T2(t_robot, t_obj)

    rotation = rel_t[0:3, 0:3]
    rel_pose_vec = sp_spatial.transform.Rotation.from_dcm(rotation).as_rotvec()
    xy = rel_t[0:2, 3]
    rel_pose_vec[0:2] = xy
    return rel_pose_vec


def get_pose_from_transform(transform):
    rotation = transform[0:3, 0:3]
    pose_vec = sp_spatial.transform.Rotation.from_dcm(rotation).as_rotvec()
    pose_vec[0:2] = transform[0:2, 3]
    return pose_vec


def get_absolute_pose_from_relative_pose(relative_pose, pose_relative_to):
    t_relative_pose = get_transform_from_pose(relative_pose, 'robot')
    t_pose_relative_to = get_transform_from_pose(pose_relative_to, 'robot')
    t_absolute = np.dot(t_pose_relative_to, t_relative_pose)
    return get_pose_from_transform(t_absolute)


def get_transform_from_pose(pose, body_type):
    pose = np.array(pose).squeeze()
    assert len(pose) == 3, "must be x,y, theta where theta is rotation around [0,0,1]"

    # rotation_mat = sp_spatial.transform.Rotation.from_rotvec([0, 0, pose[-1]]).as_dcm()

    # Got the below from https://en.wikipedia.org/wiki/3D_rotation_group, axis of rotation
    rot_angle = pose[-1]
    rotation_mat = np.array([[np.cos(rot_angle), -np.sin(rot_angle), 0],
                             [np.sin(rot_angle), np.cos(rot_angle), 0],
                             [0, 0, 1]])
    """
    try:
        assert np.all(np.isclose(rotation_mat, rotation_mat2))
    except:
        print rotation_mat
        print rotation_mat2
        import sys
        sys.exit(-1)
    """

    transformation_matrix = np.zeros((4, 4))
    transformation_matrix[0:3, 0:3] = rotation_mat
    transformation_matrix[3, 3] = 1
    transformation_matrix[0:2, 3] = pose[0:2]
    if body_type == 'robot':
        z_for_on_the_floor = 0.139183
    elif body_type == 'kinbody':
        z_for_on_the_floor = 0.389
    else:
        raise NotImplementedError

    transformation_matrix[2, 3] = z_for_on_the_floor  # this assume that the body is on the floor
    return transformation_matrix


def get_body_xytheta(body):
    if not isinstance(body, openravepy.KinBody):
        env = openravepy.RaveGetEnvironments()[0]
        body = env.GetKinBody(body)

    Tbefore = body.GetTransform()
    body_quat = get_quat(body)
    th1 = np.arccos(body_quat[0]) * 2
    th2 = np.arccos(-body_quat[0]) * 2
    th3 = -np.arccos(body_quat[0]) * 2
    quat_th1 = quat_from_angle_vector(th1, np.array([0, 0, 1]))
    quat_th2 = quat_from_angle_vector(th2, np.array([0, 0, 1]))
    quat_th3 = quat_from_angle_vector(th3, np.array([0, 0, 1]))
    if np.all(np.isclose(body_quat, quat_th1)):
        th = th1
    elif np.all(np.isclose(body_quat, quat_th2)):
        th = th2
    elif np.all(np.isclose(body_quat, quat_th3)):
        th = th3
    else:
        print "This should not happen. Check if object is not standing still"
        import pdb;
        pdb.set_trace()
    if th < 0: th += 2 * np.pi
    assert (0 <= th < 2 * np.pi)

    # set the quaternion using the one found
    set_quat(body, quat_from_angle_vector(th, np.array([0, 0, 1])))
    Tafter = body.GetTransform()
    assert (np.all(np.isclose(Tbefore, Tafter)))
    body_xytheta = np.hstack([get_point(body)[0:2], th])
    body_xytheta = body_xytheta[None, :]
    clean_pose_data(body_xytheta)
    return body_xytheta


def get_xytheta_from_transform(T):
    rotation = T[0:3, 0:3]
    xytheta = sp_spatial.transform.Rotation.from_dcm(rotation).as_rotvec()
    xytheta[0:2] = T[0:2, 3]
    return xytheta


def grab_obj(obj):
    assert len(openravepy.RaveGetEnvironments()) == 1
    env = openravepy.RaveGetEnvironments()[0]
    robot = env.GetRobot('pr2')

    robot.Grab(obj)


def release_obj():
    assert len(openravepy.RaveGetEnvironments()) == 1
    env = openravepy.RaveGetEnvironments()[0]
    robot = env.GetRobot('pr2')
    obj = robot.GetGrabbed()[0]
    robot.Release(obj)


def convert_to_kin_body(obj):
    env = openravepy.RaveGetEnvironments()[0]
    if isinstance(obj, openravepy.KinBody):
        obj = obj
    else:
        obj = env.GetKinBody(obj)
    return obj


def one_arm_pick_object(obj, pick_action):
    assert len(openravepy.RaveGetEnvironments()) == 1
    env = openravepy.RaveGetEnvironments()[0]
    robot = env.GetRobot('pr2')

    open_gripper(robot)
    manip = robot.GetManipulator('rightarm_torso')
    robot.SetActiveDOFs(manip.GetArmIndices(), DOFAffine.X | DOFAffine.Y | DOFAffine.RotationAxis, [0, 0, 1])
    pick_config = pick_action['q_goal']
    set_active_config(pick_config)
    grab_obj(obj)


def one_arm_place_object(place_action):
    assert len(openravepy.RaveGetEnvironments()) == 1
    env = openravepy.RaveGetEnvironments()[0]
    robot = env.GetRobot('pr2')

    manip = robot.GetManipulator('rightarm_torso')
    robot.SetActiveDOFs(manip.GetArmIndices(), DOFAffine.X | DOFAffine.Y | DOFAffine.RotationAxis, [0, 0, 1])
    place_config = place_action['q_goal']
    set_active_config(place_config)
    release_obj()
    open_gripper(robot)


def two_arm_place_object(place_action):
    assert len(openravepy.RaveGetEnvironments()) == 1
    env = openravepy.RaveGetEnvironments()[0]
    robot = env.GetRobot('pr2')

    try:
        place_base_pose = place_action['q_goal']
    except KeyError:
        place_base_pose = place_action['base_pose']
    leftarm_manip = robot.GetManipulator('leftarm')
    rightarm_manip = robot.GetManipulator('rightarm')

    set_robot_config(place_base_pose, robot)
    release_obj()
    set_config(robot, FOLDED_LEFT_ARM, leftarm_manip.GetArmIndices())
    set_config(robot, mirror_arm_config(FOLDED_LEFT_ARM), rightarm_manip.GetArmIndices())


def fold_arms():
    assert len(openravepy.RaveGetEnvironments()) == 1
    env = openravepy.RaveGetEnvironments()[0]
    robot = env.GetRobot('pr2')

    leftarm_manip = robot.GetManipulator('leftarm')
    rightarm_manip = robot.GetManipulator('rightarm')
    FOLDED_LEFT_ARM = [0.0, 1.29023451, 0.0, -2.12, 0.0, -0.69800004, 0.0]

    set_config(robot, FOLDED_LEFT_ARM, leftarm_manip.GetArmIndices())
    set_config(robot, mirror_arm_config(FOLDED_LEFT_ARM), rightarm_manip.GetArmIndices())


def two_arm_pick_object(obj, pick_action):
    assert len(openravepy.RaveGetEnvironments()) == 1
    env = openravepy.RaveGetEnvironments()[0]
    robot = env.GetRobot('pr2')

    if type(obj) == str or type(obj) == unicode:
        obj = robot.GetEnv().GetKinBody(obj)
    try:
        base_pose = pick_action['q_goal']
    except KeyError:
        import pdb;
        pdb.set_trace()
        base_pose = pick_action['base_pose']
    set_robot_config(base_pose, robot)

    if 'g_config' in pick_action.keys():
        g_config = pick_action['g_config']
        leftarm_manip = robot.GetManipulator('leftarm')
        rightarm_torso_manip = robot.GetManipulator('rightarm_torso')
        set_config(robot, g_config[0], leftarm_manip.GetArmIndices())
        set_config(robot, g_config[1], rightarm_torso_manip.GetArmIndices())
    grab_obj(obj)


def viewer():
    env = openravepy.RaveGetEnvironments()[0]
    env.SetViewer('qtosg')


def set_color(obj, color):
    env = openravepy.RaveGetEnvironments()[0]
    if type(obj) == str or type(obj) == unicode:
        obj = env.GetKinBody(obj)

    manipulation.bodies.bodies.set_color(obj, color)


def simulate_path(robot, path, timestep=0.001):
    for p in path:
        set_robot_config(p, robot)
        time.sleep(timestep)


def pick_distance(a1, a2, curr_obj):
    grasp_a1 = np.array(a1['grasp_params']).squeeze()
    base_a1 = clean_pose_data(np.array(a1['q_goal'])).squeeze()

    grasp_a2 = np.array(a2['grasp_params']).squeeze()
    base_a2 = clean_pose_data(np.array(a2['q_goal'])).squeeze()

    # normalize grasp distance
    grasp_max_diff = [1 / 2.356, 1., 1.]
    grasp_distance = np.sum(np.dot(abs(grasp_a1 - grasp_a2), grasp_max_diff))

    # bas_distance_max_diff = np.array([1./(2*2.51), 1./(2*2.51), 1/np.pi])
    base_distance_max_diff = np.array([1, 1, 1 / np.pi])
    base_distance = np.sum(np.dot(base_conf_diff(base_a1, base_a2), base_distance_max_diff))

    # base distance more important the grasp
    return grasp_distance + 2 * base_distance


def base_conf_diff(x, y):
    base_diff = abs(x - y)

    # This base diff computation is problematic.
    #   If the value is negative, then it will always be smaller than np.pi
    #
    th_diff = base_diff[-1]

    if x[-1] < 0:
        x[-1] += 2 * np.pi
    if y[-1] < 0:
        y[-1] += 2 * np.pi
    try:
        assert 0 <= x[-1] < 2 * np.pi
        assert 0 <= y[-1] < 2 * np.pi
    except:
        raise AssertionError, 'Base conf needs to be between [0,2pi]'
    base_diff[-1] = min(np.abs(th_diff), 2 * np.pi - np.abs(th_diff))
    return base_diff


def place_distance(a1, a2):
    base_a1 = np.array(a1['q_goal'])
    base_a1 = clean_pose_data(base_a1).squeeze()

    base_a2 = np.array(a2['q_goal'])
    base_a2 = clean_pose_data(np.array(base_a2)).squeeze()

    base_distance_max_diff = np.array([1. / 2.51, 1. / 2.51, 1 / np.pi])
    base_distance = np.sum(np.dot(base_conf_diff(base_a1, base_a2), base_distance_max_diff))

    return base_distance


def base_pose_distance(a1, a2, x_max_diff=2.51, y_max_diff=2.51):
    base_a1 = clean_pose_data(a1).squeeze()
    base_a2 = clean_pose_data(np.array(a2)).squeeze()

    base_distance_max_diff = np.array([1. / x_max_diff, 1. / y_max_diff, 1 / np.pi])
    base_distance = np.sum(np.dot(base_conf_diff(base_a1, base_a2), base_distance_max_diff))

    return base_distance


def set_rightarm_torso(config, robot=None):
    if robot is None:
        env = openravepy.RaveGetEnvironments()[0]
        robot = env.GetRobot('pr2')

    manip = robot.GetManipulator('rightarm_torso')
    robot.SetActiveDOFs(manip.GetArmIndices(), DOFAffine.X | DOFAffine.Y | DOFAffine.RotationAxis, [0, 0, 1])
    set_active_config(config, robot)


def clean_rightarm_torso_config(arm_config):
    if len(arm_config.shape) == 2:
        assert arm_config.shape[1] == 8, 'Dimension of config must be 11'
    else:
        assert len(arm_config) == 8, 'Dimension of config must be 11'
    if arm_config[5] < 0:
        arm_config[5] += 2 * np.pi

    if arm_config[-1] < 0:
        arm_config[-1] += 2 * np.pi

    if arm_config[5] > 2 * np.pi:
        arm_config[5] -= 2 * np.pi

    if arm_config[-1] > 2 * np.pi:
        arm_config[-1] -= 2 * np.pi
    return arm_config


def clean_rightarm_torso_base_pose(robot_config):
    assert len(robot_config.shape) == 1, 'robot config to be cleaned must have shape of length 1'
    clean_rightarm_torso_config(robot_config[:-3])
    clean_pose_data(robot_config[-3:])
    return robot_config


def get_rightarm_torso_config(robot=None):
    if robot is None:
        env = openravepy.RaveGetEnvironments()[0]
        robot = env.GetRobot('pr2')
    with robot:
        manip = robot.GetManipulator('rightarm_torso')
        robot.SetActiveDOFs(manip.GetArmIndices(), DOFAffine.X | DOFAffine.Y | DOFAffine.RotationAxis, [0, 0, 1])
        robot_config = robot.GetActiveDOFValues()

    robot_config = robot_config[None, :]
    clean_rightarm_torso_base_pose(robot_config[0, :])
    return robot_config


def compute_robot_xy_given_ir_parameters(portion_of_dist_to_obj, angle, t_obj, radius=PR2_ARM_LENGTH):
    dist_to_obj = radius * portion_of_dist_to_obj  # how close are you to obj?
    x = dist_to_obj * np.cos(angle)  # Relative pose to the object
    y = dist_to_obj * np.sin(angle)
    robot_wrt_o = np.array([x, y, 0, 1])
    return np.dot(t_obj, robot_wrt_o)[:-1]


def get_relative_transform_T1_wrt_T2(T1, T2):
    return np.dot(np.linalg.inv(T2), T1)


def get_relative_transform_body1_wrt_body2(body1, body2):
    return get_relative_transform_T1_wrt_T2(body1.GetTransform(), body2.GetTransform())


def subtract_pose2_from_pose1(pose1, pose2):
    pose1 = np.array(pose1).squeeze()
    pose2 = np.array(pose2).squeeze()
    diff = pose1 - pose2
    need_correction = abs(diff[-1]) > np.pi
    if need_correction and diff[-1] < 0:
        diff[-1] = 2 * np.pi + diff[-1]
    elif need_correction and diff[-1] > 0:
        diff[-1] = 2 * np.pi - diff[-1]
    return diff


def compute_ir_parameters_given_robot_xy(robot_xytheta, obj_xytheta, radius=PR2_ARM_LENGTH):
    robot_xy = robot_xytheta[0:2]
    obj_xy = obj_xytheta[0:2]
    dist_to_obj = np.linalg.norm(robot_xy - obj_xy)

    """
    robot_x_wrt_obj = robot_xy[0]-obj_xy[0] # Why is this not the case? Because the coordinate frame might not be defined at the center
    robot_y_wrt_obj = robot_xy[1]-obj_xy[1]

    robot_wrt_o = np.array([robot_x_wrt_obj, robot_y_wrt_obj, 0, 1])
    recovered = np.dot(obj.GetTransform(), robot_wrt_o)[:-1]
    """
    t_robot = get_transform_from_pose(robot_xytheta, 'robot')
    t_obj = get_transform_from_pose(obj_xytheta, 'kinbody')

    robot_xy_wrt_o = get_relative_transform_T1_wrt_T2(t_robot, t_obj)[:-2, 3]
    robot_x_wrt_obj = robot_xy_wrt_o[0]
    robot_y_wrt_obj = robot_xy_wrt_o[1]
    angle = np.arccos(abs(robot_x_wrt_obj / dist_to_obj))
    if robot_x_wrt_obj < 0 < robot_y_wrt_obj:
        angle = np.pi - angle
    elif robot_x_wrt_obj < 0 and robot_y_wrt_obj < 0:
        angle += np.pi
    elif robot_x_wrt_obj > 0 > robot_y_wrt_obj:
        angle = -angle

    portion_of_dist_to_obj = dist_to_obj / radius

    return portion_of_dist_to_obj, angle


def get_pick_base_pose_and_grasp_from_pick_parameters(obj, pick_parameters, obj_xyth=None):
    if not isinstance(obj, openravepy.KinBody):
        env = openravepy.RaveGetEnvironments()[0]
        obj = env.GetKinBody(obj)
    assert len(pick_parameters) == 6
    grasp_params = pick_parameters[0:3]
    ir_params = pick_parameters[3:]

    pick_base_pose = get_absolute_pick_base_pose_from_ir_parameters(ir_params, obj, obj_xyth)
    return grasp_params, pick_base_pose


def get_absolute_pick_base_pose_from_ir_parameters(ir_parameters, obj, obj_xyth):
    t_obj = obj.GetTransform()
    portion_of_dist_to_obj, base_angle, angle_offset = ir_parameters[0], ir_parameters[1], ir_parameters[2]

    pick_base_pose = compute_robot_xy_given_ir_parameters(portion_of_dist_to_obj, base_angle, t_obj)

    if obj_xyth is None:
        obj_xyth = get_body_xytheta(obj)

    obj_xy, robot_xy = obj_xyth.squeeze()[:-1], pick_base_pose[0:2]
    angle_to_be_set = compute_angle_to_be_set(obj_xy, robot_xy)
    pick_base_pose[-1] = angle_to_be_set + angle_offset

    return pick_base_pose


def get_ir_parameters_from_robot_obj_poses(robot_xyth, obj_xyth):
    robot_xyth = robot_xyth.squeeze()
    obj_xyth = obj_xyth.squeeze()
    obj_xy = obj_xyth.squeeze()[0:2]
    robot_xy = robot_xyth.squeeze()[:-1]
    robot_th = robot_xyth.squeeze()[-1]

    angle_to_be_set = compute_angle_to_be_set(obj_xy, robot_xy)
    facing_angle_offset = robot_th - angle_to_be_set
    while facing_angle_offset > 30. / 180 * np.pi:
        facing_angle_offset -= 2 * np.pi
    while facing_angle_offset < -30. / 180 * np.pi:
        facing_angle_offset += 2 * np.pi
    portion, base_angle = compute_ir_parameters_given_robot_xy(robot_xyth, obj_xyth)
    """
    recovered_robot_xyth = get_absolute_pick_base_pose_from_ir_parameters([portion, base_angle, facing_angle_offset],
                                                                          obj,
                                                                          obj_xyth)
    recovered_robot_xyth = clean_pose_data(recovered_robot_xyth)
    robot_xyth = clean_pose_data(robot_xyth)
    assert np.all(np.isclose(recovered_robot_xyth, robot_xyth.squeeze()))
    """
    return portion, base_angle, facing_angle_offset


def encode_pose_with_sin_and_cos_angle(pose):
    if isinstance(pose, list):
        pose = np.array(pose)
    pose = pose.reshape((3,))
    x = pose[0]
    y = pose[1]
    th = pose[2]
    sin_th_cos_th = encode_angle_in_sin_and_cos(th)
    return np.hstack([x, y, sin_th_cos_th])


def decode_pose_with_sin_and_cos_angle(pose):
    if isinstance(pose, list):
        pose = np.array(pose)
    pose = pose.reshape((4,))
    x = pose[0]
    y = pose[1]
    sin_th_cos_th = pose[2:]
    th = decode_sin_and_cos_to_angle(sin_th_cos_th)
    return np.hstack([x, y, th])


def get_global_pose_from_relative_pose_to_body(body, rel_robot_pose):
    if not isinstance(body, openravepy.KinBody):
        env = openravepy.RaveGetEnvironments()[0]
        body = env.GetKinBody(body)
    t_body = body.GetTransform()
    t_pose = get_transform_from_pose(rel_robot_pose, 'robot')
    t_pose_global = get_xytheta_from_transform(np.dot(t_body, t_pose))

    return t_pose_global


def encode_angle_in_sin_and_cos(angle):
    return np.array([np.sin(angle), np.cos(angle)])


def decode_sin_and_cos_to_angle(encoding):
    return np.arctan2(encoding[0], encoding[1])


def pick_parameter_distance(obj, param1, param2):
    grasp_params1, pick_base_pose1 = get_pick_base_pose_and_grasp_from_pick_parameters(obj, param1)
    grasp_params2, pick_base_pose2 = get_pick_base_pose_and_grasp_from_pick_parameters(obj, param2)

    base_pose_distance = se2_distance(pick_base_pose1, pick_base_pose2, 1, 1)
    grasp_distance = np.linalg.norm(grasp_params2 - grasp_params1)

    c1 = 2
    c2 = 1
    distance = c1 * base_pose_distance + c2 * grasp_distance
    return distance


def cart2pol(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return (rho, phi)


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)


def se2_distance(base_a1, base_a2, c1, c2):
    base_a1 = base_a1.squeeze()
    base_a2 = base_a2.squeeze()

    x1, y1 = pol2cart(1, base_a1[-1])
    x2, y2 = pol2cart(1, base_a2[-1])

    angle_distance = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    base_distance = np.linalg.norm(base_a1[0:2] - base_a2[0:2])

    distance = c1 * base_distance + c2 * angle_distance
    return distance


def are_base_confs_close_enough(q1, q2, xy_threshold, th_threshold):
    diff = base_conf_diff(q1, q2)
    th_threshold = th_threshold * np.pi / 180.0
    assert diff[-1] < 180.0
    if np.linalg.norm(diff[0:2]) < xy_threshold and diff[-1] < th_threshold:
        return True
    else:
        return False


def convert_base_pose_to_se2(base_pose):
    base_pose = base_pose.squeeze()
    a, b = pol2cart(1, base_pose[-1])
    x, y = base_pose[0], base_pose[1]
    return np.array([x, y, a, b])


def convert_se2_to_base_pose(basepose_se2):
    basepose_se2 = basepose_se2.squeeze()

    phi = cart2pol(basepose_se2[2], basepose_se2[3])
    return np.array([basepose_se2[0], basepose_se2[1], phi])


def place_parameter_distance(param1, param2, c1=1):
    return se2_distance(param1, param2, c1, 0)


def get_place_domain(region):
    if type(region) is str:
        if 'loading_region' == region:
            domain = np.array([[-0.7, -8.55, 0.], [4.3, -4.85, 6.28318531]])
        elif 'home_region' == region:
            domain = np.array([[-1.75, -3.16, 0], [5.25, 3.16, 6.28318531]])
        elif 'rectangular_packing_box1_region' == region:
            domain = np.array([[2.07094348, 0.06758759, 0.], [2.76927535, 0.78367914, 6.28318531]])
        else:
            raise NotImplementedError
    else:
        box = np.array(region.box)
        x_range = np.array([[box[0, 0]], [box[0, 1]]])
        y_range = np.array([[box[1, 0]], [box[1, 1]]])
        th_range = np.array([[0], [2 * np.pi]])
        domain = np.hstack([x_range, y_range, th_range])
    return domain


def get_pick_domain():
    portion_domain = [[0.4], [0.9]]
    base_angle_domain = [[0], [2 * np.pi]]
    facing_angle_domain = [[-30 * np.pi / 180.0], [30 * np.pi / 180]]
    base_pose_domain = np.hstack([portion_domain, base_angle_domain, facing_angle_domain])

    # grasp params: 45-180, 0.5-1, 0.1-0.9
    grasp_param_domain = np.array([[45 * np.pi / 180, 0.5, 0.1], [np.pi, 1, 0.9]])
    domain = np.hstack([grasp_param_domain, base_pose_domain])
    return domain


def set_obj_xyztheta(xyztheta, obj):
    set_point(obj, xyztheta[0:-1])
    set_obj_xytheta(np.hstack([xyztheta[0:2], xyztheta[-1]]), obj)


def get_body_with_name(obj_name):
    env = openravepy.RaveGetEnvironments()[0]
    return env.GetKinBody(obj_name)


def randomly_place_region(body, region, n_limit=None):
    env = openravepy.RaveGetEnvironments()[0]
    if env.GetKinBody(get_name(body)) is None:
        env.Add(body)
    orig = get_body_xytheta(body)
    # for _ in n_limit:
    i = 0
    while True:
        set_quat(body, quat_from_z_rot(uniform(0, 2 * PI)))
        aabb = aabb_from_body(body)
        cspace = region.cspace(aabb)
        if cspace is None: continue
        set_point(body, np.array([uniform(*range) for range in cspace] + [
            region.z + aabb.extents()[2] + BODY_PLACEMENT_Z_OFFSET]) - aabb.pos() + get_point(body))
        if not body_collision(env, body):
            return

        if n_limit is not None:
            i += 1
            if i >= n_limit:
                set_obj_xytheta(orig, body)
                return


class CustomStateSaver:
    def __init__(self, env):
        objects_in_env = env.GetBodies()
        self.env_id = 1

        self.robot_name = 'pr2'
        robot = env.GetRobot(self.robot_name)
        self.object_poses = {o.GetName(): get_body_xytheta(o) for o in objects_in_env}
        self.object_poses = {}
        for o in objects_in_env:
            xyz = get_point(o)
            xytheta = get_body_xytheta(o)
            xyztheta = np.hstack([xyz, xytheta.squeeze()[-1]])
            self.object_poses[o.GetName()] = xyztheta
            # todo set xyztheta

        self.robot_base_pose = get_body_xytheta(robot)
        self.robot_dof_values = robot.GetDOFValues()
        self.is_holding = len(robot.GetGrabbed()) > 0
        if self.is_holding:
            self.held_object = robot.GetGrabbed()[0].GetName()
        else:
            self.held_object = None

    def Restore(self):
        assert len(openravepy.RaveGetEnvironments()) == 1
        env = openravepy.RaveGetEnvironment(self.env_id)
        robot = env.GetRobot(self.robot_name)

        currently_holding = len(robot.GetGrabbed()) > 0
        if currently_holding:
            held_obj = robot.GetGrabbed()[0]
            release_obj()

        for obj_name, obj_pose in zip(self.object_poses.keys(), self.object_poses.values()):
            set_obj_xyztheta(obj_pose, env.GetKinBody(obj_name))
        set_robot_config(self.robot_base_pose, robot)
        robot.SetDOFValues(self.robot_dof_values)

        if self.is_holding:
            held_obj = env.GetKinBody(self.held_object)
            grab_obj(held_obj)

        # print "After-restoration"
