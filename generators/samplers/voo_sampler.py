from sampler import Sampler
import numpy as np
import time
from gtamp_utils import utils


class VOOSampler(Sampler):
    def __init__(self, target_obj, target_region, explr_p, infeasible_action_value, policy=None):
        Sampler.__init__(self, policy, target_region)
        self.explr_p = explr_p
        self.k_nearest_neighbor_to_best_point = None
        self.best_evaled_x = None
        self.infeasible_action_value = infeasible_action_value

        # todo there is pick_distance and place_distance function in the gtamp_utils
        def distance_fn(x, y):
            pick_dist = utils.pick_parameter_distance(target_obj, x[0:6], y[0:6])
            place_dist = utils.place_parameter_distance(x[6:], y[6:])
            return pick_dist + place_dist

        self.distance_fn = distance_fn
        self.distance_fn = lambda x, y: np.linalg.norm(x - y)

    def sample(self, evaled_x, evaled_y):
        rnd = np.random.random()
        is_sample_from_best_v_region = (rnd < 1 - self.explr_p) and len(evaled_x) > 1 and max(
            evaled_y) > self.infeasible_action_value
        if is_sample_from_best_v_region:
            x = self.sample_from_best_voronoi_region(evaled_x, evaled_y)
        else:
            x = self.sample_from_uniform()
        return x

    def choose_next_point(self, evaled_x, evaled_y):
        return self.sample(evaled_x, evaled_y)

    def sample_from_normal_centered_at_point(self, point, counter):
        # There are multiple ways to improve the performance of this
        # variance = max - point
        possible_max = (self.domain[1] - point) / counter
        possible_min = (self.domain[0] - point) / counter
        possible_values = np.max(np.vstack([np.abs(possible_max), np.abs(possible_min)]), axis=0)
        new_x = np.random.normal(point, possible_values)

        # make sure it is within the boundary
        counter2 = counter
        while np.any(new_x > self.domain[1]) or np.any(new_x < self.domain[0]):
            possible_max = (self.domain[1] - point) / counter2  # / np.exp(counter2)
            possible_min = (self.domain[0] - point) / counter2  # / np.exp(counter2)
            possible_values = np.max(np.vstack([np.abs(possible_max), np.abs(possible_min)]), axis=0)
            new_x = np.random.normal(point, possible_values)
            counter2 += 1
        return new_x

    def get_best_x(self, evaled_x, evaled_y):
        if self.best_evaled_x is None:
            best_evaled_x_idxs = np.argwhere(evaled_y == np.amax(evaled_y))
            best_evaled_x_idxs = best_evaled_x_idxs.reshape((len(best_evaled_x_idxs, )))
            best_evaled_x_idx = np.random.choice(best_evaled_x_idxs)
            best_evaled_x = evaled_x[best_evaled_x_idx]
            best_evaled_x = best_evaled_x
        else:
            best_evaled_x = self.best_evaled_x
        return best_evaled_x

    def get_other_points(self, evaled_x):
        if self.k_nearest_neighbor_to_best_point is None:
            other_points = evaled_x
        else:
            other_points = self.k_nearest_neighbor_to_best_point
        return other_points

    def get_dist_to_other_points(self, new_x, best_x, other_points):
        dists = []
        for other in other_points:
            if np.all(other == best_x):
                continue
            dists.append(self.distance_fn(other, new_x))
        return np.array(dists)

    def sample_from_best_voronoi_region(self, evaled_x, evaled_y):
        best_x = self.get_best_x(evaled_x, evaled_y)
        other_points = self.get_other_points(evaled_x)

        curr_closest_dist = np.inf
        for counter in range(10):
            new_x = self.sample_from_normal_centered_at_point(best_x, counter)

            dist_to_best_point = self.distance_fn(new_x, best_x)
            dists_to_other_points = self.get_dist_to_other_points(new_x, best_x, other_points)
            is_new_x_closest_to_the_best_x = dist_to_best_point < np.min(dists_to_other_points)

            if dist_to_best_point < curr_closest_dist:
                curr_closest_dist = dist_to_best_point
                curr_closest_pt = new_x

            if is_new_x_closest_to_the_best_x:
                break

        return curr_closest_pt

    def sample_from_uniform(self):
        dim_parameters = self.domain.shape[-1]
        domain_min = self.domain[0]
        domain_max = self.domain[1]
        return np.random.uniform(domain_min, domain_max, (1, dim_parameters)).squeeze()
