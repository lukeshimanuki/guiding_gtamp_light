
from data_traj import extract_individual_example as make_predictable_form
from gnn import GNN

import tensorflow as tf
import numpy as np


class PaPGNN(GNN):
    def __init__(self, num_entities, num_node_features, num_edge_features, config=None, entity_names=None, n_regions=2):
        self.n_regions = n_regions
        GNN.__init__(self, num_entities, num_node_features, num_edge_features, config, entity_names)

        self.num_entities = num_entities
        self.activation = 'relu'
        self.config = config
        self.top_k = config.top_k
        self.weight_file_name = self.create_weight_file_name()
        self.optimizer = self.create_optimizer(config.optimizer, config.lr)
        self.num_node_features = num_node_features
        self.node_input, self.edge_input, self.action_input, self.op_input, self.cost_input = \
            self.create_inputs(num_entities, num_node_features, num_edge_features, dim_action=[1])
        self.n_node_features = num_node_features

    def create_inputs(self, num_entities, num_node_features, num_edge_features, dim_action):
        num_regions = self.n_regions
        node_shape = (num_entities, num_node_features)
        edge_shape = (num_entities, num_entities, num_regions, num_edge_features)
        dim_action = (num_entities, num_regions)

        knodes = tf.keras.Input(shape=list(node_shape), name='nodes')
        kedges = tf.keras.Input(shape=list(edge_shape), name='edges')
        kactions = tf.keras.Input(shape=list(dim_action), dtype=tf.int32, name='actions')
        koperators = tf.keras.Input(shape=[1], dtype=tf.int32)
        kcosts = tf.keras.Input(shape=[1], dtype=tf.int32)

        return knodes, kedges, kactions, koperators, kcosts


    def create_weight_file_name(self):
        filedir = './learn/q-function-weights/'
        filename = "Q_weight_"
        filename += '_'.join(arg + "_" + str(getattr(self.config, arg)) for arg in [
            'n_msg_passing',
            # 'diff_weight_msg_passing',
            'mse_weight',
            'optimizer',
            # 'batch_size',
            'seed',
            'lr',
            'operator',
            'n_layers',
            'n_hidden',
            'top_k',
            'num_train',
            # 'num_test',
            # 'val_portion',
            'use_region_agnostic',
            'loss',
        ])
        filename += '.hdf5'
        print "Config:", filename
        return filedir + filename

    def load_weights(self):
        """
        self.weight_file_name = './learn/q-function-weights/trained_with_rsc_used_in_corl_submission/' \
                                'Q_weight_n_msg_passing_1_mse_weight_1.0_optimizer_' \
                                'adam_seed_%d_lr_0.0001_operator_two_arm_pick_two_arm_place_n_layers_2_n_hidden_32' \
                                '_top_k_1_num_train_5000_loss_%s.hdf5' % (self.config.seed, self.config.loss)
        """
        print "Loading weight", self.weight_file_name
        self.loss_model.load_weights(self.weight_file_name)

    def create_msg_computaton_layers(self, input, num_latent_features, name, n_layers):
        n_node_features = self.num_node_features

        n_dim_last_layer = num_latent_features if self.config.n_msg_passing == 0 else n_node_features
        if n_layers == 1:
            h = tf.keras.layers.Conv3D(n_dim_last_layer, kernel_size=(1, 1, 1),
                                       kernel_initializer=self.config.weight_initializer,
                                       bias_initializer=self.config.weight_initializer, name=name + "_0",
                                       activation='linear')(input)
            # since this represents, roughly, the value of an entity,
            # it should range from -inf to inf
            # Perhaps I could use tanh?
        else:
            h = tf.keras.layers.Conv3D(num_latent_features, kernel_size=(1, 1, 1),
                                       kernel_initializer=self.config.weight_initializer,
                                       bias_initializer=self.config.weight_initializer,
                                       name=name + "_0",
                                       activation=self.activation)(input)
            idx = -1
            for idx in range(n_layers - 2):
                h = tf.keras.layers.Conv3D(num_latent_features, kernel_size=(1, 1, 1),
                                           kernel_initializer=self.config.weight_initializer,
                                           bias_initializer=self.config.weight_initializer,
                                           name=name + "_" + str(idx + 1),
                                           activation=self.activation)(h)
            h = tf.keras.layers.Conv3D(n_dim_last_layer, kernel_size=(1, 1, 1),
                                       kernel_initializer=self.config.weight_initializer,
                                       bias_initializer=self.config.weight_initializer,
                                       name=name + "_" + str(idx + 2),
                                       activation='linear')(h)
        return h

    def create_sender_dest_edge_concatenation_lambda_layer(self):
        def concatenate_src_dest_edge_fcn(src_node_tensor, dest_node_tensor, edge_tensor):
            n_entities = self.num_entities
            src_repetitons = [1, 1, n_entities, 1]  # repeat in the columns
            repeated_srcs = tf.tile(tf.expand_dims(src_node_tensor, -2), src_repetitons)
            dest_repetitons = [1, n_entities, 1, 1]  # repeat in the rows
            repeated_dests = tf.tile(tf.expand_dims(dest_node_tensor, 1), dest_repetitons)

            src_dest_concatenated = tf.concat([repeated_srcs, repeated_dests], axis=-1)
            repetitions = [1, 1, 1, self.n_regions, 1]
            repeated_src_dest_concatenated = tf.tile(tf.expand_dims(src_dest_concatenated, -2), repetitions)

            all_concat = tf.concat([repeated_src_dest_concatenated, edge_tensor], axis=-1)

            # this should be of size n_data, n_entities, n_entities, dim_node
            # edge is of size n_data, n_entities, n_entities, n_region, dim_edge
            return all_concat

        concat_layer = tf.keras.layers.Lambda(lambda args: concatenate_src_dest_edge_fcn(*args), name='concat')
        return concat_layer

    def create_concat_model_for_verification(self):
        # create concatenation, msg computation, and aggregation layers
        concat_lambda_layer = self.create_sender_dest_edge_concatenation_lambda_layer()
        concat_layer = concat_lambda_layer([self.node_input, self.node_input, self.edge_input])

        inputs = [self.node_input, self.edge_input, self.action_input]
        self.concat_model_verifier = self.make_model(inputs, concat_layer, 'concat_model')

    def create_msg_computation_model(self, num_latent_features, name, n_layers):
        n_entities = self.num_entities
        num_regions = self.n_regions
        place_holder_input = tf.keras.Input(shape=(n_entities, n_entities, num_regions, num_latent_features * 3))
        h = self.create_msg_computaton_layers(place_holder_input, num_latent_features, name, n_layers)
        h_model = self.make_model(place_holder_input, h, 'msg_computation')
        return h_model

    def create_value_function_layers(self, config):
        msg_aggregation_layer = self.create_graph_network_layers_with_same_msg_passing_network(config)
        value_layer = tf.keras.layers.Conv2D(1, kernel_size=(1, 1),
                                             kernel_initializer=self.config.weight_initializer,
                                             bias_initializer=self.config.weight_initializer,
                                             name='value_layer',
                                             activation='linear')(msg_aggregation_layer)
        n_regions = self.n_regions

        def compute_q(values, actions):
            values = tf.squeeze(values)
            actions = tf.cast(actions, dtype=tf.float32)
            q_value = actions * values
            q_value = tf.reduce_sum(tf.reduce_sum(q_value, axis=-1), axis=-1)
            # q_value = tf.reduce_sum(tf.reduce_sum(values,axis=-1),axis=-1)
            return q_value

        q_layer = tf.keras.layers.Lambda(lambda args: compute_q(*args))
        q_layer = q_layer([value_layer, self.action_input])

        # testing purpose
        inputs = [self.node_input, self.edge_input, self.action_input]
        self.value_model = self.make_model(inputs, value_layer, 'value_model')
        self.aggregation_model = self.make_model(inputs, msg_aggregation_layer, 'value_aggregation')
        return q_layer, value_layer

    def change_sender_dest_into_shape_cancatable_to_edge(self):
        def layer_defn(sender_r1, dest_r1):
            n_entities = self.num_entities
            src_repetitons = [1, 1, n_entities, 1]  # repeat in the columns
            repeated_srcs = tf.tile(tf.expand_dims(sender_r1, -2), src_repetitons)
            dest_repetitons = [1, n_entities, 1, 1]  # repeat in the rows
            repeated_dests = tf.tile(tf.expand_dims(dest_r1, 1), dest_repetitons)
            src_dest_concatenated = tf.concat([repeated_srcs, repeated_dests], axis=-1)
            src_dest_concatenated = tf.expand_dims(src_dest_concatenated, -2)
            return src_dest_concatenated

        layer = tf.keras.layers.Lambda(lambda args: layer_defn(*args), name='')
        return layer

    def create_graph_network_layers_with_same_msg_passing_network(self, config):
        num_latent_features = config.n_hidden
        num_layers = config.n_layers
        same_model_for_sender_and_dest = config.same_vertex_model

        # create networks for vertices
        if same_model_for_sender_and_dest:
            vertex_model = self.create_multilayer_model(self.node_input, num_latent_features, 'vertex', num_layers)
            vertex_network = vertex_model(self.node_input)
        else:
            sender_model = self.create_multilayer_model(self.node_input, num_latent_features, 'src', num_layers)
            dest_model = self.create_multilayer_model(self.node_input, num_latent_features, 'dest', num_layers)
            sender_network = sender_model(self.node_input)
            dest_network = dest_model(self.node_input)

        # create edge networks
        edge_model = self.create_multilayer_model(self.edge_input, num_latent_features, 'edge', num_layers)
        edge_network = edge_model(self.edge_input)

        # create concatenation, msg computation, and aggregation layers
        concat_lambda_layer = self.create_sender_dest_edge_concatenation_lambda_layer()

        msg_model = self.create_msg_computation_model(num_latent_features, 'msgs', num_layers)
        aggregation_lambda_layer = self.create_aggregation_lambda_layer()

        if same_model_for_sender_and_dest:
            concat_layer = concat_lambda_layer([vertex_network, vertex_network, edge_network])
        else:
            concat_layer = concat_lambda_layer([sender_network, dest_network, edge_network])

        # The msg model has the same number of dimensions as the sender model
        msg_network = msg_model(concat_layer)
        msg_aggregation_layer = aggregation_lambda_layer(msg_network)  # aggregates msgs from neighbors
        # for testing purpose; delete it later
        inputs = [self.node_input, self.edge_input, self.action_input]
        self.first_msg_agg = self.make_model(inputs, msg_aggregation_layer, 'first_msg_agg')
        # rounds of msg passing
        for i in range(config.n_msg_passing):
            if same_model_for_sender_and_dest:
                vertex_network = vertex_model(msg_aggregation_layer)
                concat_layer = concat_lambda_layer([vertex_network, vertex_network, edge_network])
            else:
                if self.config.use_region_agnostic:
                    region_agnostic_msg_value = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=2), name='region_agnostic')
                    # You are just taking the aggregated messages at the goal node.
                    # Would the same thing happen if I use index 1?
                    val = region_agnostic_msg_value(msg_aggregation_layer)
                    sender_network = sender_model(val)
                    dest_network = dest_model(val)
                    concat_layer = concat_lambda_layer([sender_network, dest_network, edge_network])
                else:
                    # r1_msg_value is the set of values aggregated for moving object into a particular region
                    r1_msg_value = tf.keras.layers.Lambda(lambda x: x[:, :, 0, :], name='r1_msgs')
                    r2_msg_value = tf.keras.layers.Lambda(lambda x: x[:, :, 1, :], name='r2_msgs')
                    msg_val_r1 = r1_msg_value(msg_aggregation_layer)
                    msg_val_r2 = r2_msg_value(msg_aggregation_layer)
                    # val = region_agnostic_msg_value(msg_aggregation_layer)
                    sender_r1 = sender_model(msg_val_r1)
                    dest_r1 = dest_model(msg_val_r1)
                    sender_r2 = sender_model(msg_val_r2)
                    dest_r2 = dest_model(msg_val_r2)
                    change_sender_dest_into_shape_cancatable_to_edge_layer = self.change_sender_dest_into_shape_cancatable_to_edge()
                    src_dest_r1 = change_sender_dest_into_shape_cancatable_to_edge_layer([sender_r1, dest_r1])
                    src_dest_r2 = change_sender_dest_into_shape_cancatable_to_edge_layer([sender_r2, dest_r2])

                    def concat_layer_defn(r1_val, r2_val, edge_val):
                        r1_r2_concat = tf.concat([r1_val, r2_val], axis=-2)
                        concat_layer_fn = tf.concat([r1_r2_concat, edge_val], axis=-1)
                        return concat_layer_fn

                    concat_layer_fn = tf.keras.layers.Lambda(lambda args: concat_layer_defn(*args), name='concat2')
                    concat_layer = concat_layer_fn([src_dest_r1, src_dest_r2, edge_network])
                """
                region_agnostic_msg_value = tf.keras.layers.Lambda(lambda x: x[:, :, 0, :], name='region_agnostic')
                val = region_agnostic_msg_value(msg_aggregation_layer)
                sender_network = sender_model(val)
                dest_network = dest_model(val)
                concat_layer = concat_lambda_layer([sender_network, dest_network, edge_network])
                """
            msg_network = msg_model(concat_layer)
            msg_aggregation_layer = aggregation_lambda_layer(msg_network)  # aggregates msgs from neighbors
            # todo it is saturating here when I use relu?

        self.second_msg_agg = self.make_model(inputs, msg_aggregation_layer, 'second_msg_agg')
        self.msg_model = self.make_model(inputs, msg_network, 'msg_model')
        self.concat_model = self.make_model(inputs, concat_layer, 'concat_model')
        if same_model_for_sender_and_dest:
            self.sender_model = vertex_model  # sender_model
            self.dest_model = vertex_model  # dest_model
        else:
            self.sender_model = sender_model  # sender_model
            self.dest_model = dest_model  # dest_model

        self.edge_model = edge_model
        return msg_aggregation_layer

    def predict_with_raw_input_format(self, nodes, edges, actions):
        nodes = np.array(nodes)
        edges = np.array(edges)
        actions = np.array(actions)
        if len(nodes.shape) == 2:
            nodes = nodes[None, :]
        if len(edges.shape) == 3:
            edges = edges[None, :]
        if len(actions.shape) == 0:
            actions = actions[None]
        return self.q_model.predict([nodes[:, :, :], edges, actions])

    def create_loss_model(self, q_layer, value_layer):
        n_regions = self.n_regions
        n_entities = self.num_entities

        def compute_rank_loss(alt_msgs, target_msg, costs):
            alt_msgs = tf.squeeze(alt_msgs)
            alt_msgs = tf.reshape(alt_msgs, [-1, n_regions * n_entities])
            sorted_top_k_q_vals = tf.nn.top_k(alt_msgs, self.top_k, sorted=True)[0]  # sorts it in descending order
            min_of_top_k_q_val = sorted_top_k_q_vals[:, -1]
            q_delta = target_msg - min_of_top_k_q_val
            action_ranking_cost = 1.0 - q_delta
            hinge_loss = tf.reduce_mean(tf.maximum(tf.cast(0., tf.float32), action_ranking_cost))
            # hinge_loss = action_ranking_cost
            return hinge_loss

        def compute_mse_loss(msgs, target_msg, costs):
            return tf.losses.mean_squared_error(target_msg, -costs)

        # https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
        def compute_dql_loss(msgs, target_msg, costs):
            msgs = tf.squeeze(msgs)
            msgs = tf.reshape(msgs, [-1, n_regions * n_entities])
            target_msg = tf.squeeze(target_msg)
            costs = tf.squeeze(costs)

            gamma = .99
            final = tf.cast(costs[:-1] <= 2, tf.float32)
            reward = -2. + 101. * final
            q_s_a = target_msg[:-1]
            q_sprime_aprime = tf.reduce_max(msgs, -1)[1:]
            y = reward + gamma * q_sprime_aprime * (1 - final)
            return tf.losses.mean_squared_error(q_s_a, y)

        def get_alt_msgs(msgs, actions):
            values = tf.squeeze(msgs)
            actions = tf.cast(actions, dtype=tf.float32)
            num_entities = self.num_entities

            # attention = tf.squeeze(1 - tf.one_hot(action_idxs, num_entities, dtype=tf.float32))

            alt_msgs = (1 - actions) * values

            # For some reason, I cannot use boolean_mask in the loss functions.
            # I am speculating that it has to do with the fact that the loss functions receive a batch of data,
            # instead of a single data point
            # This is a workaround for making tf.math.top_k to not choose the action seen in the planning experience
            neg_inf_attention_on_plan_actions = actions * -2e32
            alt_msgs = alt_msgs + neg_inf_attention_on_plan_actions

            return alt_msgs

        alt_msgs = tf.keras.layers.Lambda(lambda args: get_alt_msgs(*args))
        alt_msg_layer = alt_msgs([value_layer, self.action_input])

        if self.config.loss == 'mse':
            loss_layer = tf.keras.layers.Lambda(
                lambda args: compute_mse_loss(*args))
        else:
            loss_layer = tf.keras.layers.Lambda(
                lambda args: compute_dql_loss(*args) if self.config.loss == 'dql' else (
                        compute_rank_loss(*args) + self.config.mse_weight * compute_mse_loss(*args)))

        loss_layer = loss_layer([alt_msg_layer, q_layer, self.cost_input])
        loss_inputs = [self.node_input, self.edge_input, self.action_input, self.cost_input]
        loss_model = tf.keras.Model(loss_inputs, loss_layer)
        loss_model.compile(loss=lambda _, loss_as_pred: loss_as_pred, optimizer=self.optimizer)

        # testing purpose
        inputs = [self.node_input, self.edge_input, self.action_input]
        self.alt_msg_layer = self.make_model(inputs, alt_msg_layer, 'alt_msg')
        return loss_model

    def make_raw_format(self, state, op_skeleton):
        nodes, edges, action, _ = make_predictable_form(state, op_skeleton)
        nodes = nodes[None, :]
        edges = edges[None, :]
        action = action[None, :]
        return nodes[:, :, 6:], edges, action

    def predict(self, state, op_skeleton):
        nodes, edges, action = self.make_raw_format(state, op_skeleton)
        val = self.predict_with_raw_input_format(nodes, edges, action)
        return val
