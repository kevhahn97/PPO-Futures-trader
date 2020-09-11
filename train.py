import os
from typing import Optional

from stable_baselines.common import make_vec_env
from stable_baselines.common.callbacks import BaseCallback

from stable_baselines.common.policies import FeedForwardPolicy, MlpPolicy, LstmPolicy, CnnPolicy, CnnLstmPolicy, MlpLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, DQN

import tensorflow as tf
import numpy as np

from custom_env import FutureTradingEnv


def get_ext(input_config):
    def ext(input_tensor, **kwargs):
        split_list = list()
        for key, item in sorted(input_config.items()):
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


def get_geometric_average_and_cummulative_profit(model, env: FutureTradingEnv, env_name: str, code: Optional[str] = None, verbose: int = 1):
    env.reset(code=code)
    num_dates = len(env.parsed_date_list)
    profit_list = list()
    if verbose == 1:
        print(f'\n---------- start simulating {env_name} -----------')
    for date_idx in range(num_dates):
        obs = env.reset(code=code, date_idx=date_idx)
        done = False
        while not done:
            if True:
                action, _ = model.predict(obs)
                obs, _, done, _ = env.step(action)
            else:
                action, _ = model.predict(np.asarray([obs]))
                obs, _, done, _ = env.step(action[0])
            if done:
                daily_profit = env.get_daily_profit()
                profit_list.append(daily_profit)
                if verbose == 1:
                    print(f'date: {env.current_date}\tprofit: {daily_profit * 100:.3f}%')
    profits = np.asarray(profit_list, dtype=np.float64)
    profits += 1
    cummulative_profit = profits.prod()
    geometric_average_profit = cummulative_profit ** (1.0 / len(profits))
    cummulative_profit -= 1
    geometric_average_profit -= 1
    if verbose == 1:
        print(f'cummulative profit for {len(profits)} trading days: {cummulative_profit * 100}%')
        print(f'geometric average profit for {len(profits)} trading days: {geometric_average_profit * 100}%')
        print(f'---------- done simulating {env_name} -----------\n')
    return geometric_average_profit, cummulative_profit


class ProfitCallback(BaseCallback):
    def __init__(self, train_env, valid_env, epoch: int, checkpointing_metric='geometric_average', checkpoint_dir: Optional[str] = None, verbose=0):
        self.is_tb_set = False
        self.train_env = train_env
        self.valid_env = valid_env
        self.checkpointing_metric = checkpointing_metric
        self.best_valid_profit = -99999
        self.checkpoint_dir = checkpoint_dir
        self.epoch = epoch
        super(ProfitCallback, self).__init__(verbose)

    def _on_training_end(self) -> None:
        if not self.is_tb_set:
            self.is_tb_set = True
        train_geometric_average_profit, train_cummulative_profit = get_geometric_average_and_cummulative_profit(model=self.model, env=self.train_env, env_name='train_env', code='122630', verbose=self.verbose)
        valid_geometric_average_profit, valid_cummulative_profit = get_geometric_average_and_cummulative_profit(model=self.model, env=self.valid_env, env_name='valid_env', code='122630', verbose=self.verbose)
        summary = tf.Summary(value=[
            tf.Summary.Value(tag='profit/train_geo_avg_profit', simple_value=train_geometric_average_profit),
            tf.Summary.Value(tag='profit/train_cum_profit', simple_value=train_cummulative_profit),
            tf.Summary.Value(tag='profit/valid_geo_avg_profit', simple_value=valid_geometric_average_profit),
            tf.Summary.Value(tag='profit/valid_cum_profit', simple_value=valid_cummulative_profit),
        ])
        self.locals['writer'].add_summary(summary, self.epoch)
        valid_checkpointing_metric = valid_geometric_average_profit if self.checkpointing_metric == 'geometric_average' else valid_cummulative_profit
        if self.best_valid_profit < valid_checkpointing_metric:
            self.best_valid_profit = valid_checkpointing_metric
            if self.verbose == 1:
                print(f'New best {self.checkpointing_metric}: {self.best_valid_profit * 100} at epoch #{self.epoch}')
                if self.checkpoint_dir is not None:
                    checkpoint_path = os.path.join(checkpoint_dir, str(self.epoch))
                    self.model.save(checkpoint_path)
                    print(f'Saving to {checkpoint_path}')

    def _on_step(self) -> bool:
        self.training_env.env_method('print_profit')
        return True


n_envs = 1
n_steps = 378
input_config = {'1분': {'N': 128, 'columns': ['open', 'high', 'low', 'close']}}
env = make_vec_env(FutureTradingEnv, n_envs, env_kwargs=dict(db_file_path='C:\\Users\\hanseungho\\stocks\\Kiwoom_datareader\\db\\200810_025124_주식분봉_122630.db', start_date=20190801, end_date=20200131, input_config=input_config))

version = 'v_4_mlp_always_reward'

tensorboard_dir = os.path.join('log', 'tb')
checkpoint_dir = os.path.join('log', 'ckpt', version)
os.makedirs(tensorboard_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# model = PPO2(FeedForwardPolicy, env, verbose=1, n_steps=378, policy_kwargs=dict(feature_extraction="cnn", cnn_extractor=get_ext(input_config)))
model = PPO2(MlpPolicy, env, verbose=1, n_steps=n_steps, nminibatches=1, tensorboard_log=tensorboard_dir)
# model = PPO2(MlpLstmPolicy, env, verbose=1, n_steps=n_steps, nminibatches=1, tensorboard_log=tensorboard_dir)

train_env = FutureTradingEnv(db_file_path='C:\\Users\\hanseungho\\stocks\\Kiwoom_datareader\\db\\200810_025124_주식분봉_122630.db', start_date=20190801, end_date=20200131, input_config=input_config)
valid_env = FutureTradingEnv(db_file_path='C:\\Users\\hanseungho\\stocks\\Kiwoom_datareader\\db\\200810_025124_주식분봉_122630.db', start_date=20200131, end_date=20200801, input_config=input_config)

num_episode = 10000
steps_per_epoch = 64
for epoch in range(num_episode):
    print(f'\n***Epoch {epoch}/{num_episode} start***')
    model.learn(total_timesteps=n_steps * n_envs * steps_per_epoch, callback=ProfitCallback(train_env, valid_env, epoch=epoch, checkpointing_metric='geometric_average', checkpoint_dir=checkpoint_dir, verbose=1), reset_num_timesteps=False, tb_log_name=version)
    print(f'***Epoch {epoch}/{num_episode} done***\n')
