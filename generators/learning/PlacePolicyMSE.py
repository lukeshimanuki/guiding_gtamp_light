from PlacePolicy import PlacePolicy

import socket

if socket.gethostname() == 'lab' or socket.gethostname() == 'phaedra':
    ROOTDIR = './'
else:
    ROOTDIR = '/data/public/rw/pass.port/guiding_gtamp/'


class PlacePolicyMSE(PlacePolicy):
    def __init__(self, dim_action, dim_collision, save_folder, tau, config):
        PlacePolicy.__init__(self, dim_action, dim_collision, save_folder, tau, config)

    def construct_policy_output(self):
        raise NotImplementedError

    def construct_policy_model(self):
        raise NotImplementedError

