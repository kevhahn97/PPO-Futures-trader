import tensorflow as tf
import numpy as np
from stable_baselines.common.tf_layers import lstm, linear


def get_ext(input_config):
    def ext(input_tensor, **kwargs):
        split_list = list()
        for key, item in sorted(input_config['chart'].items()):
            N, columns = item['N'], item['columns']
            split_list.append([N * len(columns), N, len(columns)])
        split_list.extend([[1, 1, 1]] * 4)  # 4 additional features

        *charts, average_position_price, position, current_price_to_current_account_valuation, tick_to_close = tf.split(input_tensor, np.asarray(split_list)[..., 0], axis=-1)

        charts_reshaped = list()
        for chart, split in zip(charts, split_list):
            assert chart.shape[-1] == split[0]
            charts_reshaped.append(tf.reshape(chart, [-1] + split[1:]))

        chart_features = list()
        for chart in charts_reshaped:
            x = chart
            x = tf.keras.layers.Conv1D(8, 7)(x)
            x = tf.keras.layers.ReLU()(x)
            x = tf.keras.layers.AveragePooling1D()(x)
            x = tf.keras.layers.Conv1D(16, 7)(x)
            x = tf.keras.layers.ReLU()(x)
            x = tf.keras.layers.AveragePooling1D()(x)
            x = tf.keras.layers.Conv1D(32, 7)(x)
            x = tf.keras.layers.ReLU()(x)
            x = tf.keras.layers.AveragePooling1D()(x)
            x = tf.keras.layers.Conv1D(64, 7)(x)
            x = tf.keras.layers.ReLU()(x)
            # x = tf.keras.layers.AveragePooling1D()(x)
            # x = tf.keras.layers.Conv1D(64, 7)(x)
            # x = tf.keras.layers.ReLU()(x)
            x = tf.keras.layers.GlobalAveragePooling1D()(x)
            # x = tf.keras.layers.LSTM(8)(x)
            chart_features.append(x)
        features = tf.concat(chart_features + [average_position_price, position, current_price_to_current_account_valuation, tick_to_close], axis=-1)
        features = tf.keras.layers.Dense(64)(features)
        features = tf.keras.layers.ReLU()(features)
        # features = tf.keras.layers.Dense(64)(features)
        # features = tf.keras.layers.ReLU()(features)
        return features

    return ext


def chart_feature_extractor_mlp(chart):
    x = chart
    x = tf.keras.layers.Flatten()(x)
    # x = tf.keras.layers.Dense(128)(x)
    # x = tf.tanh(x)
    x = tf.keras.layers.Dense(8)(x)
    x = tf.tanh(x)

    return x


def chart_feature_extractor_lstm(chart):
    x = chart
    # x = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNLSTM(units=64, kernel_initializer='orthogonal', return_sequences=True))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNLSTM(units=8, kernel_initializer='orthogonal', return_sequences=False))(x)
    # x = lstm(x, tf.zeros_like(x), tf.zeros_like(x), scope='extractor', n_hidden=64)
    return x


def get_extractor(input_config):
    def _extractor(input_tensor, **kwargs):
        split_list = list()
        for key, item in sorted(input_config['chart'].items()):
            N, columns = item['N'], item['columns']
            split_list.append([N * len(columns), N, len(columns)])
        num_account_features = len(input_config['account'])
        split_list.extend([[1, 1, 1]] * num_account_features)  # additional features

        # *charts, average_position_price, position, current_price_to_current_account_valuation, tick_to_close = tf.split(input_tensor, np.asarray(split_list)[..., 0], axis=-1)
        *charts, average_position_price, position = tf.split(input_tensor, np.asarray(split_list)[..., 0], axis=-1)

        charts_reshaped = list()
        for chart, split in zip(charts, split_list):
            assert chart.shape[-1] == split[0]
            charts_reshaped.append(tf.reshape(chart, [-1] + split[1:]))

        chart_features = list()
        for chart in charts_reshaped:
            chart_features.append(chart_feature_extractor_lstm(chart))

        # account_features = [
        #     average_position_price, tf.square(average_position_price), tf.sqrt(average_position_price),
        #     position, tf.square(position),
        #     current_price_to_current_account_valuation, tf.square(current_price_to_current_account_valuation), tf.sqrt(current_price_to_current_account_valuation),
        #     tick_to_close, tf.square(tick_to_close), tf.sqrt(tick_to_close)]
        account_features = [average_position_price, position]
        skip = position
        features = tf.concat(chart_features + account_features, axis=-1)
        features = tf.keras.layers.Dense(64, kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2)))(features)
        features = tf.keras.layers.Activation('tanh')(features)
        # features = tf.keras.layers.Dropout(0.5)(features)
        # features = tf.keras.layers.Dropout(0.5)(features)
        # features = tf.keras.layers.Dense(128, kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2)))(features)
        # features = tf.keras.layers.Activation('tanh')(features)
        # features = tf.keras.layers.Dropout(0.5)(features)
        # features = tf.keras.layers.Dense(128, kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2)))(features)
        # features = tf.keras.layers.ReLU()(features)
        # features = tf.tanh(linear(features, scope='extractor', n_hidden=128, init_scale=np.sqrt(2)))
        # features = tf.tanh(linear(features, scope='extractor', n_hidden=128, init_scale=np.sqrt(2)))
        return features, skip

    return _extractor
