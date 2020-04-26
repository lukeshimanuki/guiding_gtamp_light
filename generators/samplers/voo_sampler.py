from sampler import Sampler
import numpy as np
import time

class VOOSampler(Sampler):
    def __init__(self, target_region, explr_p, infeasible_action_value, policy=None):
        Sampler.__init__(self, policy, target_region)
        self.explr_p = explr_p
        self.k_nearest_neighbor_to_best_point = None
        self.best_evaled_x = None
        self.infeasible_action_value = infeasible_action_value

        # todo there is pick_distance and place_distance function in the gtamp_utils
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
        possible_max = (self.domain[1] - point) / np.exp(counter)
        possible_min = (self.domain[0] - point) / np.exp(counter)
        possible_values = np.max(np.vstack([np.abs(possible_max), np.abs(possible_min)]), axis=0)
        new_x = np.random.normal(point, possible_values)

        # make sure it is within the boundary
        counter2 = counter
        while np.any(new_x > self.domain[1]) or np.any(new_x < self.domain[0]):
            possible_max = (self.domain[1] - point) / np.exp(counter2)
            possible_min = (self.domain[0] - point) / np.exp(counter2)
            possible_values = np.max(np.vstack([np.abs(possible_max), np.abs(possible_min)]), axis=0)
            new_x = np.random.normal(point, possible_values)
            counter2 += 1
        return new_x

    def sample_from_best_voronoi_region(self, evaled_x, evaled_y):
        best_dist = np.inf
        other_dists = np.array([-1])
        counter = 0

        if self.best_evaled_x is None:
            best_evaled_x_idxs = np.argwhere(evaled_y == np.amax(evaled_y))
            best_evaled_x_idxs = best_evaled_x_idxs.reshape((len(best_evaled_x_idxs, )))
            best_evaled_x_idx = np.random.choice(best_evaled_x_idxs)
            best_evaled_x = evaled_x[best_evaled_x_idx]
            best_evaled_x = best_evaled_x
        else:
            best_evaled_x = self.best_evaled_x
        curr_closest_dist = np.inf

        while np.any(best_dist > other_dists):
            if self.k_nearest_neighbor_to_best_point is None:
                other_points = evaled_x
            else:
                other_points = self.k_nearest_neighbor_to_best_point
            new_x = self.sample_from_normal_centered_at_point(best_evaled_x, counter)
            best_dist = self.distance_fn(new_x, best_evaled_x)
            other_dists = np.array([self.distance_fn(other, new_x) for other in other_points])
            counter += 1
            if best_dist < curr_closest_dist:
                curr_closest_dist = best_dist
                curr_closest_pt = new_x

        return curr_closest_pt

    def sample_from_uniform(self):
        dim_parameters = self.domain.shape[-1]
        domain_min = self.domain[0]
        domain_max = self.domain[1]
        return np.random.uniform(domain_min, domain_max, (1, dim_parameters)).squeeze()
