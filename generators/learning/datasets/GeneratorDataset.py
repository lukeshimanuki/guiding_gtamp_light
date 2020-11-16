from torch.utils.data import Dataset

import pickle
import numpy as np
import torch
from generators.learning.utils.data_processing_utils import get_processed_poses_from_state, \
    get_processed_poses_from_action

from gtamp_utils import utils
import os


class GivenDataset(Dataset):
    def __init__(self, actions, konf_obsts, poses):
        self.actions = actions
        self.konf_obsts = konf_obsts
        self.poses = poses

    def __len__(self):
        return len(self.konf_obsts)

    def __getitem__(self, idx):
        data = {
            'konf_obsts': self.konf_obsts[idx],
            'poses': self.poses[idx],
            'actions': self.actions[idx],
        }
        return data


class GeneratorDataset(Dataset):
    def __init__(self, config, use_filter, is_testing):
        self.use_filter = use_filter
        self.is_testing = is_testing
        self.config = config
        # if self.config.train_type == 'w':
        self.konf_obsts, self.poses, self.actions, self.labels, self.dists_to_goal, self.all_object_robot_poses \
            = self.get_data()

    def get_cache_file_name(self, action_data_mode):
        state_mode = self.config.state_mode
        action_type = self.config.atype
        desired_region = self.config.region
        use_filter = self.use_filter
        if 'pick' in action_type:
            cache_file_name = 'hard_cache_smode_%s_amode_%s_atype_%s_num_data_%d.pkl' % (
                state_mode, action_data_mode, action_type, self.config.num_episode)
        else:
            assert use_filter
            cache_file_name = 'hard_cache_smode_%s_amode_%s_atype_%s_region_%s_filtered_num_episode_%d.pkl' % (
                state_mode,
                action_data_mode,
                action_type,
                desired_region,
                self.config.num_episode)
        return cache_file_name

    def get_data_dir(self):
        if self.use_filter:
            if 'one_arm' in self.config.domain:
                data_dir = 'planning_experience/processed/one_arm_mover/n_objs_pack_1/sahs/uses_rrt/' \
                           'sampler_trajectory_data/includes_n_in_way/includes_vmanip/'
                return [data_dir]
            else:
                data_dir_4_objs = 'planning_experience_phaedra/planning_experience/processed/two_arm_mover/' \
                                  'n_objs_pack_4/sahs/uses_rrt/' \
                                  'sampler_trajectory_data/includes_n_in_way/includes_vmanip/'
                data_dir_1_obj = 'planning_experience_phaedra/planning_experience/processed/two_arm_mover/' \
                                 'n_objs_pack_1/sahs/uses_rrt/' \
                                 'sampler_trajectory_data/includes_n_in_way/includes_vmanip/'
                return [data_dir_4_objs, data_dir_1_obj]
        else:
            raise NotImplementedError

    def we_should_skip_this_state_and_action(self, s, reward):
        action_type = self.config.atype
        desired_region = self.config.region
        use_filter = self.use_filter
        if 'place' in action_type:
            is_move_to_goal_region = s.region in s.goal_entities
            # fix this to include the one that moves the object to the goal
            if reward <= 0 and use_filter:
                return True

            if 'two_arm' in self.config.domain:
                if desired_region == 'home_region' and not is_move_to_goal_region:
                    return True

                if desired_region == 'loading_region' and is_move_to_goal_region:
                    return True
            else:
                if desired_region == 'rectangular_packing_box1_region' and not is_move_to_goal_region:
                    return True

                if desired_region == 'center_shelf_region' and is_move_to_goal_region:
                    return True

        return False

    @staticmethod
    def get_object_poses(abstract_state, manipulated_object_name):
        obj_list = [
            u'rectangular_packing_box4', u'rectangular_packing_box2', u'rectangular_packing_box3',
            u'rectangular_packing_box1', u'square_packing_box2', u'square_packing_box3', u'square_packing_box1',
            u'square_packing_box4'
        ]

        pose_based_state = abstract_state.nodes[manipulated_object_name][3:6]
        for obj_name in obj_list:
            if obj_name == manipulated_object_name:
                continue
            pose = abstract_state.nodes[obj_name][3:6]
            pose_based_state += pose
        return pose_based_state

    def get_data_from_traj_dir(self, traj_dir, action_data_mode):
        # todo fix this;
        traj_files = os.listdir(traj_dir)
        cache_file_name = self.get_cache_file_name(action_data_mode)
        if os.path.isfile(traj_dir + cache_file_name):
            print "Loading the cache file", traj_dir + cache_file_name
            f = pickle.load(open(traj_dir + cache_file_name, 'r'))
            print "Cache data loaded"
            return f

        n_episodes = 0
        all_states = []
        all_actions = []
        all_poses_ids = []
        all_labels = []
        all_dists_to_goal = []
        all_traj_all_object_and_robot_poses = []
        n_episodes_included = 0
        for traj_file_idx, traj_file in enumerate(traj_files):
            if 'pidx' not in traj_file:
                print 'not pkl file'
                continue
            pidx = int(traj_file.split('pidx_')[1].split('_')[0])
            if pidx < 40000:
                continue
            try:
                traj = pickle.load(open(traj_dir + traj_file, 'r'))
            except:
                cmd = 'python data_processing/sampler/process_planning_experience.py -n_objs_pack 1 -pidx {} ' \
                      '-domain two_arm_mover -planner greedy -absq_seed 2 -f'.format(pidx)
                os.system(cmd)
                try:
                    traj = pickle.load(open(traj_dir + traj_file, 'r'))
                except:
                    continue
            if len(traj['positive_data'] + traj['neutral_data']) == 0:
                continue

            traj_states = []
            traj_poses_ids = []  # includes only current obj, goal obj, and robot poses
            traj_all_object_and_robot_poses = []
            traj_actions = []
            traj_data = traj['positive_data'] + traj['neutral_data']
            traj_temp_labels = [1] * len(traj['positive_data']) + [0] * len(traj['neutral_data'])
            traj_dists_to_goal = []

            labels = []
            assert 'two_arm' in traj_data[0]['action'].type, 'change the reward condition on region for one_arm'
            print "traj file:", traj_file
            pos_data_idx = 1
            for node, temp_label in zip(traj_data, traj_temp_labels):
                reward = (node['parent_n_in_way'] - node['n_in_way'] > 0) or \
                         (node['parent_n_in_way'] == 0 and node['n_in_way'] == 0 and
                          node['action'].discrete_parameters['place_region'] == 'home_region')

                s = node['concrete_state']
                if self.we_should_skip_this_state_and_action(s, reward):
                    continue
                else:
                    labels.append(temp_label)
                    collision_vec = s.pick_collision_vector
                    v_manip_goal = node['parent_v_manip']
                    if 'two_arm' in self.config.domain:
                        v_manip_vec = utils.convert_binary_vec_to_one_hot(v_manip_goal.squeeze()).reshape(
                            (1, 618, 2, 1))
                    else:
                        v_manip_vec = utils.convert_binary_vec_to_one_hot(v_manip_goal.squeeze()).reshape(
                            (1, 355, 2, 1))
                    state_vec = np.concatenate([collision_vec, v_manip_vec], axis=2)
                    traj_states.append(state_vec)
                    poses_from_state = get_processed_poses_from_state(s, 'absolute')
                    a = node['action_info']
                    if 'rectangular' in a['object_name']:
                        object_id = [1, 0]
                    else:
                        object_id = [0, 1]
                    poses_from_state_and_id = np.hstack([poses_from_state, object_id])
                    traj_poses_ids.append(poses_from_state_and_id)
                    traj_actions.append(get_processed_poses_from_action(s, a, action_data_mode))

                    if temp_label == 1:
                        dist_to_goal = len(traj['positive_data']) - pos_data_idx
                        pos_data_idx += 1
                        traj_dists_to_goal.append(dist_to_goal)
                    else:
                        traj_dists_to_goal.append(-999)

                    # get the object poses
                    all_object_poses = self.get_object_poses(node['abs_state'],
                                                             node['concrete_state'].abstract_action.discrete_parameters[
                                                                 'object'])
                    robot_pose = node['abs_state'].robot_pose.squeeze()
                    all_object_and_robot_poses = all_object_poses + robot_pose.tolist()
                    traj_all_object_and_robot_poses.append(np.array(all_object_and_robot_poses))

            if len(traj_poses_ids) == 0:
                continue
            all_poses_ids.append(traj_poses_ids)
            all_states.append(traj_states)
            all_actions.append(traj_actions)
            all_labels.append(labels)
            all_dists_to_goal.append(traj_dists_to_goal)
            all_traj_all_object_and_robot_poses.append(traj_all_object_and_robot_poses)

            assert len(np.hstack(all_labels)) == len(np.vstack(all_poses_ids))

            try:
                print 'n_data %d progress %d/%d n_pos %d n_neutral %d ' \
                      % (len(np.vstack(all_actions)), traj_file_idx, len(traj_files), np.sum(np.hstack(all_labels)),
                         np.sum(np.hstack(all_labels) == 0))
            except:
                import pdb;
                pdb.set_trace()

            print 'action shape', np.vstack(all_actions).shape
            n_episodes_included += 1
            print 'n_episodes included %d/%d' % (n_episodes_included, self.config.num_episode)
            if n_episodes_included >= self.config.num_episode:
                break

        all_states = np.vstack(all_states).squeeze(axis=1)
        all_actions = np.vstack(all_actions).squeeze()
        all_poses_ids = np.vstack(all_poses_ids).squeeze()
        all_labels = np.hstack(all_labels).squeeze()
        all_dists_to_goal = np.hstack(all_dists_to_goal).squeeze()
        all_traj_all_object_and_robot_poses = np.vstack(all_traj_all_object_and_robot_poses)
        pickle.dump((all_states, all_poses_ids, all_actions, all_labels, all_dists_to_goal,
                     all_traj_all_object_and_robot_poses),
                    open(traj_dir + cache_file_name, 'wb'))
        return all_states, all_poses_ids, all_actions, all_labels, all_dists_to_goal, \
               all_traj_all_object_and_robot_poses

    def load_pos_neu_data(self, action_data_mode):
        traj_dirs = self.get_data_dir()
        all_states = []
        all_actions = []
        all_poses_ids = []
        all_labels = []
        all_dists_to_goal = []
        all_all_object_robot_poses = []
        for traj_dir in traj_dirs:
            states, poses_ids, actions, labels, dists_to_goal, all_object_robot_poses = self.get_data_from_traj_dir(
                traj_dir, action_data_mode)
            all_states.append(states)
            all_actions.append(actions)
            all_poses_ids.append(poses_ids)
            all_labels.append(labels)
            all_dists_to_goal.append(dists_to_goal)
            all_all_object_robot_poses.append(all_object_robot_poses)
        all_states = np.vstack(all_states)
        all_actions = np.vstack(all_actions)
        all_poses_ids = np.vstack(all_poses_ids)
        all_labels = np.hstack(all_labels)
        all_dists_to_goal = np.hstack(all_dists_to_goal)
        all_all_object_robot_poses = np.vstack(all_all_object_robot_poses)
        return all_states, all_poses_ids, all_actions, all_labels, all_dists_to_goal, all_all_object_robot_poses

    def get_data(self):
        if self.config.atype == 'pick':
            action_data_mode = 'PICK_grasp_params_and_ir_parameters_PLACE_abs_base'
        else:
            action_data_mode = 'PICK_grasp_params_and_abs_base_PLACE_abs_base'

        states, poses, actions, labels, dists_to_goal, all_object_robot_poses = self.load_pos_neu_data(action_data_mode)

        if self.config.atype == 'pick':
            actions = actions[:, :-4]
        elif self.config.atype == 'place':
            pick_abs_poses = actions[:, 3:7]  # must swap out the q0 with the pick base pose
            poses[:, -6:-2] = pick_abs_poses
            actions = actions[:, -4:]
        else:
            raise NotImplementedError

        return states, poses, actions, labels, dists_to_goal, all_object_robot_poses

    def __len__(self):
        return len(self.konf_obsts)

    def __getitem__(self, idx):
        raise NotImplementedError


class StandardDataset(GeneratorDataset):
    def __init__(self, config, use_filter, is_testing):
        super(StandardDataset, self).__init__(config, use_filter, is_testing)
        self.konf_obsts = self.konf_obsts[self.labels == 1]
        self.poses = self.poses[self.labels == 1]
        self.actions = self.actions[self.labels == 1]
        self.dists_to_goal = self.dists_to_goal[self.labels == 1]
        self.all_object_robot_poses = self.all_object_robot_poses[self.labels == 1]

    def __getitem__(self, idx):
        data = {
            'konf_obsts': self.konf_obsts[idx],
            'poses': self.poses[idx],
            'actions': self.actions[idx],
            'dists_to_goal': self.dists_to_goal[idx],
            'all_object_robot_poses': self.all_object_robot_poses[idx]
        }
        return data


class ImportanceEstimatorDataset(GeneratorDataset):
    def __init__(self, config, use_filter, is_testing):
        super(ImportanceEstimatorDataset, self).__init__(config, use_filter, is_testing)

    def __getitem__(self, idx):
        data = {
            'konf_obsts': self.konf_obsts[idx],
            'poses': self.poses[idx],
            'actions': self.actions[idx],
            'labels': self.labels[idx]
        }
        return data
