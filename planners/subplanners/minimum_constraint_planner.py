from planners.subplanners.motion_planner import BaseMotionPlanner, ArmBaseMotionPlanner
import time


class MinimumConstraintPlanner(BaseMotionPlanner, ArmBaseMotionPlanner):
    def __init__(self, problem_env, target_object, planning_algorithm):
        BaseMotionPlanner.__init__(self, problem_env, planning_algorithm)
        if type(target_object) == str:
            self.target_object = self.problem_env.env.GetKinBody(target_object)
        else:
            self.target_object = target_object

    def approximate_minimal_collision_path(self, goal_configuration, path_ignoring_all_objects,
                                           collisions_in_path_ignoring_all_objects, cached_collisions):
        # enabled objects = all the objects except the ones that are in collision
        enabled_objects = {obj.GetName() for obj in self.problem_env.objects}
        enabled_objects -= {obj.GetName() for obj in collisions_in_path_ignoring_all_objects}

        [o.Enable(False) for o in collisions_in_path_ignoring_all_objects]

        # enable object one by one. If we can find a path with it turned off, then it is not actually in the way.
        # minimal objects in way = a set of objects that we cannot ignore
        # for other objects in collisions_in_path_ignoring_all_objects, we can ignore
        minimal_objects_in_way = []
        minimal_collision_path = path_ignoring_all_objects
        print "Approximating MCR path..."
        for obj in collisions_in_path_ignoring_all_objects:
            obj.Enable(True)
            [o.Enable(False) for o in minimal_objects_in_way]
            enabled_objects.add(obj.GetName())
            enabled_objects -= {obj.GetName() for obj in minimal_objects_in_way}
            if self.problem_env.name.find('one_arm') != -1:
                path, status = ArmBaseMotionPlanner.get_motion_plan(self, goal_configuration,
                                                                    cached_collisions=cached_collisions)
            else:
                path, status = BaseMotionPlanner.get_motion_plan(self,
                                                                 goal_configuration,
                                                                 cached_collisions=cached_collisions,
                                                                 n_iterations=[20, 50, 100, 500, 1000])
            if status != 'HasSolution':
                minimal_objects_in_way.append(obj)
            else:
                minimal_collision_path = path

        self.problem_env.enable_objects_in_region('entire_region')
        return minimal_collision_path

    def compute_path_ignoring_obstacles(self, goal_configuration):
        self.problem_env.disable_objects_in_region('entire_region')
        if self.target_object is not None:
            self.target_object.Enable(True)
        if self.problem_env.name.find('one_arm') != -1:
            path, status = ArmBaseMotionPlanner.get_motion_plan(self, goal_configuration)
        else:
            # stime = time.time()
            path, status = BaseMotionPlanner.get_motion_plan(self, goal_configuration,
                                                             n_iterations=[20, 50, 100, 500, 1000])
            # print "Motion plan time", time.time()-stime
        self.problem_env.enable_objects_in_region('entire_region')
        if path is None:
            #import pdb; pdb.set_trace()
            print('### WARNING: path ignoring obstacles not found ###')
        return path

    def get_motion_plan(self, goal_configuration, region_name='entire_region', n_iterations=None,
                        cached_collisions=None):
        path_ignoring_obstacles = self.compute_path_ignoring_obstacles(goal_configuration)
        if path_ignoring_obstacles is None:
            return None, "NoSolution"

        naive_path_collisions = self.problem_env.get_objs_in_collision(path_ignoring_obstacles, 'entire_region')
        assert not (self.target_object in naive_path_collisions)

        no_obstacle = len(naive_path_collisions) == 0
        if no_obstacle:
            return path_ignoring_obstacles, 'HasSolution'

        minimal_collision_path = self.approximate_minimal_collision_path(goal_configuration, path_ignoring_obstacles,
                                                                         naive_path_collisions, cached_collisions)
        self.problem_env.enable_objects_in_region('entire_region')
        return minimal_collision_path, "HasSolution"
