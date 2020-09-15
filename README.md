# PPO Futures trader
Optimize futures trading strategy using Proximal Policy Optimization (PPO)

WIP..

### Python env
```
conda env create -f environment.yml -n [ENV_NAME]
```

### Using trading gym env
```python
from future_trading_env_discrete import FutureTradingEnvDiscrete

input_config = {
  'chart': {
    '1분': {'N': 128, 'columns': ['open', 'high', 'low', 'close']},
    ...
  }, 
  'account': ['position', 'average_position_price', ...]
}

env = FutureTradingEnvDiscrete(
  db_file_path='./db', 
  start_date=20200101, end_date=20200830, 
  input_config=input_config, penalty=0.003, train=True)

# choose random code and date from db
env.reset()

# can specify one
env.reset(code='005930')
env.reset(date_idx=0)
env.reset(code='005930', date_idx=0)
```

### PPO
stable-baselines package: [link](https://github.com/hill-a/stable-baselines)

paper: [link](https://arxiv.org/abs/1707.06347)

OpenAI Five: [link](https://openai.com/blog/openai-five/)
