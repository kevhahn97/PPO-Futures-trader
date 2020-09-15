import glob
import os
import shutil
from typing import Optional

from stable_baselines.common import make_vec_env
from stable_baselines.common.callbacks import BaseCallback

from stable_baselines.common.policies import FeedForwardPolicy, MlpPolicy, LstmPolicy, CnnPolicy, CnnLstmPolicy, MlpLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, DQN, TRPO

from callbacks import ProfitCallback
from custom_env import FutureTradingEnv
from extractors import get_lstm_extractor
from future_trading_env_discrete import FutureTradingEnvDiscrete


def get_version(memo: str, log_base_dir='log'):
    version_list = os.listdir(log_base_dir)
    version_list = list(map(lambda x: int(x.split('_')[1]), version_list))
    version_list.sort()
    if len(version_list) == 0:
        version = 1
    else:
        version = version_list[-1] + 1
    version = f'v_{version}_{memo}'
    print(f'train version: {version}')
    return version


def backup_source(source_backup_dir):
    source_list = glob.glob('*.py') + glob.glob('*.yml') + glob.glob('*.sh')
    for source in source_list:
        target_path = os.path.join(source_backup_dir, source)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        shutil.copy(source, target_path)


# env_cls = FutureTradingEnv
env_cls = FutureTradingEnvDiscrete
n_envs = 32
nminibatches = 4
noptepochs = 2
n_steps = 377
db_file_path = 'db/200816_012555.db'
# db_file_path = 'db/200810_025124_주식분봉_122630.db'
simulation_db_file_path = 'db/200810_025124_주식분봉_122630.db'
# sim_train_code = '122630'
sim_train_code = '005930'
sim_valid_code = '122630'
input_config = {'chart': {'1분': {'N': 128, 'columns': ['open', 'high', 'low', 'close']}}, 'account': ['position', 'average_posiion']}
penalty = 0.003

train_start = 20190801
train_end = 20200131
# train_end = 20190802

valid_start = 20200203
valid_end = 20200801
# valid_end = 20200204

env = make_vec_env(env_cls, n_envs, env_kwargs=dict(db_file_path=db_file_path, start_date=train_start, end_date=train_end, input_config=input_config, penalty=penalty))

memo = 'discrete_action_ritternorm_k0_N_128_horizon_377_account_features_2_price_norm_0'
if memo is None:
    memo = input('memo: ')
version = get_version(memo=memo, log_base_dir='log')

log_dir = os.path.join('log', version)
tensorboard_dir = os.path.join(log_dir, 'tb')
checkpoint_dir = os.path.join(log_dir, 'ckpt')
backup_dir = os.path.join(log_dir, 'backup')
os.makedirs(tensorboard_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(backup_dir, exist_ok=True)

backup_source(source_backup_dir=backup_dir)

# LstmPolicy is default
# Mlp extractor
# model = PPO2(MlpLstmPolicy, env, verbose=1, n_steps=n_steps, nminibatches=nminibatches, tensorboard_log=tensorboard_dir, noptepochs=noptepochs)

# Custom extractor
model = PPO2(LstmPolicy, env, verbose=1, n_steps=n_steps, nminibatches=nminibatches, tensorboard_log=tensorboard_dir, noptepochs=noptepochs, policy_kwargs=dict(feature_extraction="cnn", cnn_extractor=get_lstm_extractor(input_config)))

# model = PPO2(MlpLstmPolicy, env, verbose=1, n_steps=n_steps, nminibatches=1, tensorboard_log=tensorboard_dir)

train_env = env_cls(db_file_path=db_file_path, start_date=train_start, end_date=train_end, input_config=input_config, penalty=penalty, train=False)
valid_env = env_cls(db_file_path=simulation_db_file_path, start_date=valid_start, end_date=valid_end, input_config=input_config, penalty=penalty, train=False)

freq = 512
# freq = 4
num_episode = 10000 * freq
model.learn(total_timesteps=n_steps * n_envs * num_episode, callback=ProfitCallback(train_env, valid_env, train_code=sim_train_code, valid_code=sim_valid_code, freq=n_steps * n_envs * freq, checkpointing_metric='geometric_average', checkpoint_dir=checkpoint_dir, version_name=version, verbose=1), reset_num_timesteps=False)
