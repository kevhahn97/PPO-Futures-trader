import warnings

from stable_baselines.common.policies import RecurrentActorCriticPolicy
import tensorflow as tf
import numpy as np
from stable_baselines.common.tf_layers import linear, lstm
from stable_baselines.common.tf_util import batch_to_seq, seq_to_batch


class MyLstmPolicy(RecurrentActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, extractor=None, n_lstm=256, reuse=False, net_arch=None, act_fun=tf.tanh, layer_norm=False, use_skip=True, **kwargs):
        # state_shape = [n_lstm * 2] dim because of the cell and hidden states of the LSTM
        super(MyLstmPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch,
                                           state_shape=(2 * n_lstm,), reuse=reuse,
                                           scale=False)
        if False:
            if layers is None:
                layers = [64, 64]
            else:
                warnings.warn("The layers parameter is deprecated. Use the net_arch parameter instead.")

            with tf.variable_scope("model", reuse=reuse):
                if extractor is not None:
                    extracted_features = extractor(self.processed_obs, **kwargs)
                else:
                    extracted_features = tf.layers.flatten(self.processed_obs)
                    for i, layer_size in enumerate(layers):
                        extracted_features = act_fun(linear(extracted_features, 'pi_fc' + str(i), n_hidden=layer_size,
                                                            init_scale=np.sqrt(2)))
                input_sequence = batch_to_seq(extracted_features, self.n_env, n_steps)
                masks = batch_to_seq(self.dones_ph, self.n_env, n_steps)
                rnn_output, self.snew = lstm(input_sequence, masks, self.states_ph, 'lstm1', n_hidden=n_lstm,
                                             layer_norm=layer_norm)
                rnn_output = seq_to_batch(rnn_output)
                value_fn = linear(rnn_output, 'vf', 1)

                self._proba_distribution, self._policy, self.q_value = \
                    self.pdtype.proba_distribution_from_latent(rnn_output, rnn_output)

            self._value_fn = value_fn

        if net_arch is None:
            net_arch = ['lstm', {'pi': [64, 64], 'vf': [64, 64]}]
        with tf.variable_scope("model", reuse=reuse):
            obs_feature = self.processed_obs

            skip = None
            if extractor is not None:
                skip, obs_feature = extractor(obs_feature, **kwargs)

            latent = tf.layers.flatten(obs_feature)

            if skip is not None and use_skip:
                latent = tf.concat([latent, skip], axis=-1)

            policy_only_layers = []  # Layer sizes of the network that only belongs to the policy network
            value_only_layers = []  # Layer sizes of the network that only belongs to the value network

            # Iterate through the shared layers and build the shared parts of the network
            lstm_layer_constructed = False
            for idx, layer in enumerate(net_arch):
                if isinstance(layer, int):  # Check that this is a shared layer
                    layer_size = layer
                    latent = act_fun(linear(latent, "shared_fc{}".format(idx), layer_size, init_scale=np.sqrt(2)))
                elif layer == "lstm":
                    if lstm_layer_constructed:
                        raise ValueError("The net_arch parameter must only contain one occurrence of 'lstm'!")
                    input_sequence = batch_to_seq(latent, self.n_env, n_steps)
                    masks = batch_to_seq(self.dones_ph, self.n_env, n_steps)
                    rnn_output, self.snew = lstm(input_sequence, masks, self.states_ph, 'lstm1', n_hidden=n_lstm,
                                                 layer_norm=layer_norm)
                    latent = seq_to_batch(rnn_output)
                    lstm_layer_constructed = True
                else:
                    assert isinstance(layer, dict), "Error: the net_arch list can only contain ints and dicts"
                    if 'pi' in layer:
                        assert isinstance(layer['pi'],
                                          list), "Error: net_arch[-1]['pi'] must contain a list of integers."
                        policy_only_layers = layer['pi']

                    if 'vf' in layer:
                        assert isinstance(layer['vf'],
                                          list), "Error: net_arch[-1]['vf'] must contain a list of integers."
                        value_only_layers = layer['vf']
                    break  # From here on the network splits up in policy and value network

            # Build the non-shared part of the policy-network
            latent_policy = latent
            for idx, pi_layer_size in enumerate(policy_only_layers):
                if pi_layer_size == "lstm":
                    raise NotImplementedError("LSTMs are only supported in the shared part of the policy network.")
                assert isinstance(pi_layer_size, int), "Error: net_arch[-1]['pi'] must only contain integers."
                latent_policy = act_fun(
                    linear(latent_policy, "pi_fc{}".format(idx), pi_layer_size, init_scale=np.sqrt(2)))

            # Build the non-shared part of the value-network
            latent_value = latent
            for idx, vf_layer_size in enumerate(value_only_layers):
                if vf_layer_size == "lstm":
                    raise NotImplementedError("LSTMs are only supported in the shared part of the value function "
                                              "network.")
                assert isinstance(vf_layer_size, int), "Error: net_arch[-1]['vf'] must only contain integers."
                latent_value = act_fun(
                    linear(latent_value, "vf_fc{}".format(idx), vf_layer_size, init_scale=np.sqrt(2)))

            if not lstm_layer_constructed:
                raise ValueError("The net_arch parameter must contain at least one occurrence of 'lstm'!")

            self._value_fn = linear(latent_value, 'vf', 1)

            if skip is not None and use_skip:
                latent_policy = tf.concat([latent_policy, skip], axis=-1)

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(latent_policy, latent_value)
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            return self.sess.run([self.deterministic_action, self.value_flat, self.snew, self.neglogp],
                                 {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask})
        else:
            return self.sess.run([self.action, self.value_flat, self.snew, self.neglogp],
                                 {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask})

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask})
