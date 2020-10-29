import os
from typing import Optional, Union

from stable_baselines.common.callbacks import BaseCallback

import tensorflow as tf
import numpy as np

from custom_env import FutureTradingEnv
from future_trading_env_discrete import FutureTradingEnvDiscrete


def _make_env_fns_and_get_obs(env: Union[FutureTradingEnv, FutureTradingEnvDiscrete], date_idx_list, code):
    env_fns = list()
    obs_list = list()
    for date_idx in date_idx_list:
        _env = env.copy()
        obs_list.append(_env.reset(code, date_idx))
        env_fns.append(lambda: _env)
    return env_fns, obs_list


def get_geometric_average_and_cummulative_profit(model, env: Union[FutureTradingEnv, FutureTradingEnvDiscrete], env_name: str, version_name: str, code: Optional[str] = None, verbose: int = 1):
    env.reset(code=code)
    profit_list = list()
    if verbose == 1:
        print(f'\n---------- start simulating {env_name} -----------')
    for date_idx in env.simulatable_date_idx_list:
        obs = env.reset(code=code, date_idx=date_idx)
        done = False
        state = None
        while not done:
            if True:
                action, state = model.predict(np.stack([obs.tolist()] * model.n_envs), state, deterministic=True)
                obs, _, done, _ = env.step(action[0])
            else:
                action, _ = model.predict(np.asarray([obs]))
                obs, _, done, _ = env.step(action[0])
            if done:
                daily_profit = env.get_daily_profit()
                print(env.tick_to_close)
                profit_list.append(daily_profit)
                if verbose == 1:
                    print(version_name)
                    env.render()
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
    def __init__(self, train_env, valid_env, train_code: str, valid_code: str, version_name: str, freq: Optional[int] = None, checkpointing_metric='geometric_average', checkpoint_dir: Optional[str] = None, verbose=0):
        self.is_tb_set = False
        self.train_env = train_env
        self.valid_env = valid_env
        self.train_code = train_code
        self.valid_code = valid_code
        self.checkpointing_metric = checkpointing_metric
        self.best_valid_profit = -99999
        self.checkpoint_dir = checkpoint_dir
        if freq is None:
            freq = self.model.n_steps * self.model.n_envs * 64
        self.freq = freq
        self.version_name = version_name
        super(ProfitCallback, self).__init__(verbose)

    def _on_training_end(self) -> None:
        pass

    def _on_step(self) -> bool:
        # self.training_env.env_method('print_profit')
        if self.num_timesteps % self.freq == 0:
            version = self.num_timesteps // self.freq - 1
            if not self.is_tb_set:
                self.is_tb_set = True
            train_geometric_average_profit, train_cummulative_profit = get_geometric_average_and_cummulative_profit(model=self.model, env=self.train_env, env_name='train_env', version_name=self.version_name, code=self.train_code, verbose=self.verbose)
            valid_geometric_average_profit, valid_cummulative_profit = get_geometric_average_and_cummulative_profit(model=self.model, env=self.valid_env, env_name='valid_env', version_name=self.version_name, code=self.valid_code, verbose=self.verbose)
            summary = tf.Summary(value=[
                tf.Summary.Value(tag='profit/train_geo_avg_profit', simple_value=train_geometric_average_profit),
                tf.Summary.Value(tag='profit/train_cum_profit', simple_value=train_cummulative_profit),
                tf.Summary.Value(tag='profit/valid_geo_avg_profit', simple_value=valid_geometric_average_profit),
                tf.Summary.Value(tag='profit/valid_cum_profit', simple_value=valid_cummulative_profit),
            ])
            self.locals['writer'].add_summary(summary, version)
            valid_checkpointing_metric = valid_geometric_average_profit if self.checkpointing_metric == 'geometric_average' else valid_cummulative_profit
            if self.best_valid_profit < valid_checkpointing_metric:
                self.best_valid_profit = valid_checkpointing_metric
                if self.verbose == 1:
                    print(f'New best {self.checkpointing_metric}: {self.best_valid_profit * 100} at epoch #{version}')
                    if self.checkpoint_dir is not None:
                        checkpoint_path = os.path.join(self.checkpoint_dir, str(version))
                        self.model.save(checkpoint_path)
                        print(f'Saving to {checkpoint_path}')
        return True
