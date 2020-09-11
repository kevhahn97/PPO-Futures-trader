import sqlite3
import math
from typing import Optional

import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np


class FutureTradingEnv(gym.Env):
    """
    Description:
        Future chart data is prepared for given term. Not only Future charts, any charts of an asset, such as ETF or stock, can also be used.

    Source:
        This environment is solely created by Seungho Han

    Observation:
        Every observation includes
            1. past chart data
            2. account info
            3. position info
            4. time info (hint for market closing time)

        spaces.Dict({
            "chart": spaces.Tuple((spaces.Box(shape=(N, 4)), spaces.Box(shape=(N, 4)))),
            "account_valuation": spaces.Box(1),
            "average_position_price": spaces.Box(1),
            "position": spaces.Box(1),
            "current_price_to_current_account_valuation": spaces.Box(1),
            "tick_to_close": spaces.Box(1),
        })

        chart: Box(N, 4)
        Num	    name    Min     Max
        0	    open    0.      inf
        1	    high    0.      inf
        2	    low     0.      inf
        3	    close   0.      inf

       #account_valuation: Box(1) (current_account_valuation / initial_account_valuation)
       #Num	    name    Min     Max
       #0	    open    0.      inf

        average_position_price: Box(1) (average_position_price / current_price)

        position: Box(1) (average_position_price * num_contracts / current_account_valuation)

        current_price_to_current_account_valuation: Box(1) (current_price / current_account_valuation)

    Actions:
        Every action sets position for the asset which is expressed by a float number of interval [-1, 1]
        Type: Box(1)
        Num	    Action      Min     Max
        0	    Position    -1.     1.

    Reward:
        Reward is given when position is liquidated, by the value representing earned reward by that contract
        Actual reward is calculated as (realzied profit / initial account valuation)
        At terminating step, reward is calculated regarding the action is 0.
        TODO: force to make 0. action at terminating step by giving large penalty reward when not

    Starting State:
        Every env.reset() call sets
            1. random chart data (etc. random code, random date)
            2. random initial account valuation (random value of interval [asset price x 3, asset price x 1000])
            3. random position (random value of interval [100, 200])

    Episode Termination:
        Market closes. Actually, at 15:19:00, when 1518 candle is completed, episode ends with liquidating all position.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, db_file_path, start_date, end_date, input_config: dict, penalty=0.003):
        self.ticks_per_trading_day = 378  # 15:19 - 09:01 = total 378 ticks. hence, it has 378 steps of size 1/377
        self.db_file_path = db_file_path
        self.start_date = start_date
        self.end_date = end_date
        self.input_config = input_config

        self.db_connection = None
        self.db_cursor = None
        self._init_db()

        # set on reset
        self.current_code = None
        self.parsed_date_list = None
        self.current_date = None
        self.current_date_idx = None
        self.initial_account_valuation = None
        self.checkpoint_account_valuation = None

        # states (changes every step)
        self.current_chart = None
        self.current_price = None
        self.average_position_price = None
        self.position = None
        self.current_cash = None
        self.tick_to_close = None
        self.penalty = penalty

        chart_features = 0
        # chart_obs_dict = dict()
        for key, item in input_config.items():
            N = item['N']
            columns = item['columns']
            # chart_obs_dict[key] = spaces.Box(low=0, high=np.inf, shape=(N, len(columns)))
            chart_features += N * len(columns)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,))
        # self.observation_space = spaces.Dict({
        #     'chart': spaces.Dict(chart_obs_dict),
        #     # 'account_valuation': spaces.Box(low=0, high=np.inf, shape=1),
        #     'average_position_price': spaces.Box(low=0, high=np.inf, shape=(1,)),
        #     'position': spaces.Box(low=-1, high=1, shape=(1,)),
        #     'current_price_to_current_account_valuation': spaces.Box(low=0, high=1, shape=(1,)),
        #     'tick_to_close': spaces.Box(low=0, high=1, shape=(1,)),
        # })
        # self.observation_space = spaces.Box(low=-np.finfo(np.float32).max, high=np.finfo(np.float32).max, shape=(chart_features+4, ))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(chart_features + 4,))

        self.seed()
        # self.viewer = None
        # self.state = None
        #
        # self.steps_beyond_done = None

    @property
    def current_position_valuation(self):
        return self.position * (self.current_price if self.position > 0 else (2 * self.average_position_price - self.current_price) * -1)

    @property
    def normalized_position(self):
        return self.current_position_valuation / self.current_account_valuation * (-1 if self.position < 0 else 1)

    @property
    def current_account_valuation(self):
        return self.current_cash + self.current_position_valuation

    def _init_db(self):
        self.db_connection = sqlite3.connect(self.db_file_path)
        self.db_cursor = self.db_connection.cursor()
        self.db_cursor.execute("""select name from sqlite_master where type='table'""")
        table_list = [x[0] for x in self.db_cursor.fetchall()]
        code_dict = dict()
        for table in table_list:
            asset_type, code, tick = table.split('_')
            for ic in self.input_config:
                if ic == tick:
                    if not code_dict.get(code):
                        code_dict[code] = dict()
                        code_dict[code]['asset_type'] = asset_type
                    code_dict[code][tick] = True

        self.validated_code_dict = dict()
        for code, data in code_dict.items():
            if sum(list(map(lambda data_key: data_key in self.input_config or data_key == 'asset_type', data))) == len(self.input_config) + 1:
                self.validated_code_dict[code] = data['asset_type']

    def _normalize_chart(self, chart, columns):
        if 'volume' in columns:
            raise NotImplementedError
        assert self.current_price == chart[-1][columns.index('close')]
        return chart / self.current_price

    def _observe(self):
        flattened_feature_list = list()
        # chart_obs_dict = dict()
        for key, item in sorted(self.input_config.items()):
            N, columns = item['N'], item['columns']
            column_indices = list()
            for column in columns:
                column_indices.append(self.current_chart[key]['columns'].index(column))
            date_slice = slice(self.current_chart[key]['current_idx'] - (N - 1), self.current_chart[key]['current_idx'] + 1)
            chart = self.current_chart[key]['data'][date_slice, column_indices]
            # chart_obs_dict[key] = self._normalize_chart(chart, columns)
            flattened_feature_list.append(self._normalize_chart(chart, columns).flatten())
        average_position_price = self.average_position_price / self.current_price
        position = self.normalized_position
        current_price_to_current_account_valuation = self.current_price / self.current_account_valuation
        tick_to_close = self.tick_to_close / (self.ticks_per_trading_day - 1)
        flattened_feature_list.append([average_position_price, position, current_price_to_current_account_valuation, tick_to_close])
        """return {
                    'chart': chart_obs_dict,
                    'average_position_price': self.average_position_price / self.current_price,
                    'position': self.normalized_position,
                    'current_price_to_current_account_valuation': self.current_price / self.current_account_valuation,
                    'tick_to_close': self.tick_to_close / (self.ticks_per_trading_day - 1)
                }"""
        return np.concatenate(flattened_feature_list, axis=0)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        # get states
        # make action: position, current_cash, average_position_price, current_account_valuation changes
        # calculate reward for current action
        # move 1 tick: current_chart index, current_price, average_position_price, current_account_valuation, tick_to_close changes

        done = self.tick_to_close == 0

        is_profit_realized = False

        new_normalized_position = action[0] if not done else 0.
        old_normalized_position = self.normalized_position
        if old_normalized_position >= 0:
            if old_normalized_position < new_normalized_position:  # more long
                step = self.current_price / self.current_account_valuation

                old_position = self.position
                old_average_position_price = self.average_position_price
                old_current_cash = self.current_cash

                num_contracts = int(round((new_normalized_position - old_normalized_position) / step, 1))
                assert num_contracts >= 0
                contract_valuation = num_contracts * self.current_price
                assert contract_valuation >= 0

                new_position = old_position + num_contracts
                if new_position == 0:
                    new_average_position_price = 0
                else:
                    assert new_position > 0
                    new_average_position_price = (old_position * old_average_position_price + contract_valuation) / new_position
                new_current_cash = old_current_cash - abs(contract_valuation) * (1 + 0)  # 0 penalty for buying

                self.average_position_price = new_average_position_price
                self.position = new_position
                self.current_cash = new_current_cash
            elif old_normalized_position == new_normalized_position:  # hold
                pass
            elif 0 <= new_normalized_position < old_normalized_position:  # liquidate long
                step = self.current_price / self.current_account_valuation

                old_position = self.position
                old_average_position_price = self.average_position_price
                old_current_cash = self.current_cash

                num_contracts = int(round((new_normalized_position - old_normalized_position) / step, 1))
                assert num_contracts <= 0
                contract_valuation = num_contracts * self.current_price
                assert contract_valuation <= 0

                new_position = old_position + num_contracts
                if new_position == 0:
                    new_average_position_price = 0
                else:
                    assert new_position > 0
                    new_average_position_price = old_average_position_price
                new_current_cash = old_current_cash + abs(contract_valuation) * (1 - self.penalty)

                self.average_position_price = new_average_position_price
                self.position = new_position
                self.current_cash = new_current_cash

                is_profit_realized = num_contracts != 0
            elif new_normalized_position < 0:  # liquidate all long and switch to short
                step = self.current_price / self.current_account_valuation

                old_position = self.position
                old_average_position_price = self.average_position_price
                old_current_cash = self.current_cash

                num_contracts = int(round((0 - old_normalized_position) / step, 1))
                assert num_contracts <= 0
                contract_valuation = num_contracts * self.current_price
                assert contract_valuation <= 0

                assert old_position + num_contracts == 0
                new_current_cash = old_current_cash + abs(contract_valuation) * (1 - self.penalty)
                # liquidate done

                # short
                num_short_contracts = int(round((new_normalized_position - 0) / step, 1))
                assert num_short_contracts <= 0
                short_contract_valuation = num_short_contracts * self.current_price
                assert short_contract_valuation <= 0

                new_position = num_short_contracts
                if new_position == 0:
                    new_average_position_price = 0
                else:
                    assert new_position < 0
                    new_average_position_price = abs(short_contract_valuation / new_position)
                new_current_cash = new_current_cash - abs(short_contract_valuation) * (1 + 0)  # 0 penalty for buy

                self.average_position_price = new_average_position_price
                self.position = new_position
                self.current_cash = new_current_cash

                is_profit_realized = num_contracts != 0
        else:
            if new_normalized_position < old_normalized_position:  # more short
                step = self.current_price / self.current_account_valuation

                old_position = self.position
                old_average_position_price = self.average_position_price
                old_current_cash = self.current_cash

                num_contracts = int(round((new_normalized_position - old_normalized_position) / step, 1))
                assert num_contracts <= 0
                contract_valuation = num_contracts * self.current_price
                assert contract_valuation <= 0

                new_position = old_position + num_contracts
                if new_position == 0:
                    new_average_position_price = 0
                else:
                    assert new_position < 0  # and old_position < 0
                    new_average_position_price = (abs(old_position) * old_average_position_price + abs(contract_valuation)) / abs(new_position)
                    # same as
                    # new_average_position_price = (old_position * old_average_position_price + contract_valuation) / new_position
                new_current_cash = old_current_cash - abs(contract_valuation) * (1 + 0)  # 0 penalty for buy

                self.average_position_price = new_average_position_price
                self.position = new_position
                self.current_cash = new_current_cash
            elif old_normalized_position == new_normalized_position:  # hold
                pass
            elif old_normalized_position < new_normalized_position <= 0:  # liquidate short
                step = (2 * self.average_position_price - self.current_price) / self.current_account_valuation

                old_position = self.position
                old_average_position_price = self.average_position_price
                old_current_cash = self.current_cash

                num_contracts = int(round((new_normalized_position - old_normalized_position) / step, 1))
                assert num_contracts >= 0
                contract_valuation = num_contracts * (2 * old_average_position_price - self.current_price)
                assert contract_valuation >= 0

                new_position = old_position + num_contracts
                if new_position == 0:
                    new_average_position_price = 0
                else:
                    assert new_position < 0
                    new_average_position_price = old_average_position_price
                new_current_cash = old_current_cash + abs(contract_valuation) * (1 - self.penalty)

                self.average_position_price = new_average_position_price
                self.position = new_position
                self.current_cash = new_current_cash

                is_profit_realized = num_contracts != 0
            elif 0 < new_normalized_position:  # liquidate all short and switch to long
                step = (2 * self.average_position_price - self.current_price) / self.current_account_valuation

                old_position = self.position
                old_average_position_price = self.average_position_price
                old_current_cash = self.current_cash

                num_contracts = int(round((0 - old_normalized_position) / step, 1))
                assert num_contracts >= 0
                contract_valuation = num_contracts * (2 * old_average_position_price - self.current_price)
                assert contract_valuation >= 0

                assert old_position + num_contracts == 0
                new_current_cash = old_current_cash + abs(contract_valuation) * (1 - self.penalty)
                # liquidate done

                # long
                long_step = self.current_price / self.current_account_valuation

                num_long_contracts = int(round((new_normalized_position - 0) / long_step, 1))
                assert num_long_contracts >= 0
                long_contract_valuation = num_long_contracts * self.current_price
                assert long_contract_valuation >= 0

                new_position = num_long_contracts
                if new_position == 0:
                    new_average_position_price = 0
                else:
                    assert new_position > 0
                    new_average_position_price = abs(long_contract_valuation / new_position)
                new_current_cash = new_current_cash - abs(long_contract_valuation) * (1 + 0)  # 0 penalty for buy

                self.average_position_price = new_average_position_price
                self.position = new_position
                self.current_cash = new_current_cash

                is_profit_realized = num_contracts != 0
        if is_profit_realized or True:
            # reward
            short_term_reward = (self.current_account_valuation / self.checkpoint_account_valuation) - 1.0
            long_term_reward = (self.current_account_valuation / self.initial_account_valuation) - 1.0
            reward = short_term_reward + long_term_reward * 0
            reward *= 100
            self.checkpoint_account_valuation = self.current_account_valuation
        else:
            reward = 0

        if not done:
            # tick
            # evaluate current_price
            self.tick_to_close -= 1
            for ic in self.input_config:
                if ic == '1분':
                    self.current_chart[ic]['current_idx'] += 1
                    new_idx = self.current_chart[ic]['current_idx']
                    columns = self.current_chart[ic]['columns']
                    self.current_price = self.current_chart[ic]['data'][new_idx][columns.index('close')]

        return self._observe(), reward, done, {}

    def _get_chart_data(self, tick, start_date_time, end_date_time):
        self.db_cursor.execute(f'select * from {self.validated_code_dict[self.current_code]}_{self.current_code}_{tick} where date >= ? and date <= ? order by date', (start_date_time, end_date_time))
        data = np.asarray(self.db_cursor.fetchall())
        columns = [x[0] for x in self.db_cursor.description]
        return data, columns

    def _parse_date_from_1min_candle(self):
        required_past_dates = 0
        for ic in self.input_config:
            if ic == '1분':
                required_past_dates = max(required_past_dates, math.ceil(self.input_config[ic]['N'] / 378))
            elif ic == '60분':
                required_past_dates = max(required_past_dates, math.ceil(self.input_config[ic]['N'] / 7))
        query_start_date_time = self._add_time_to_date(self.start_date, 000000)
        query_end_date_time = self._add_time_to_date(self.end_date, 999999)
        data, columns = self._get_chart_data('1분', query_start_date_time, query_end_date_time)
        date_idx = columns.index('date')
        self.parsed_date_list = sorted(list(set((data[:, date_idx] // 1000000).tolist())))
        self.simulatable_date_idx_list = range(required_past_dates, len(self.parsed_date_list))

    def _add_time_to_date(self, date: int, time: int) -> int:
        d, t = f'{date:08}', f'{time:06}'
        assert len(f'{date:08}') == 8, 'date must be 8 digits integer'
        assert len(f'{time:06}') == 6, 'time must be 6 digits integer'
        return int(d + t)

    def _prepare_1min_chart(self):
        ic = '1분'
        required_past_dates = math.ceil(self.input_config[ic]['N'] / 378)

        query_start_date_time = self._add_time_to_date(self.parsed_date_list[self.current_date_idx - required_past_dates], 000000)
        query_end_date_time = self._add_time_to_date(self.parsed_date_list[self.current_date_idx - 1], 999999)
        past_data, _ = self._get_chart_data(ic, query_start_date_time, query_end_date_time)
        assert past_data.shape[0] > 0, f'No data found for table: {self.validated_code_dict[self.current_code]}_{self.current_code}_{ic}. date: {query_start_date_time} ~ {query_end_date_time}'

        query_start_date_time = self._add_time_to_date(self.current_date, 000000)
        query_end_date_time = self._add_time_to_date(self.current_date, 999999)
        data, columns = self._get_chart_data(ic, query_start_date_time, query_end_date_time)
        assert data.shape[0] > 0, f'No data found for table: {self.validated_code_dict[self.current_code]}_{self.current_code}_{ic}. date: {query_start_date_time} ~ {query_end_date_time}'

        open_idx = len(past_data)
        close_idx = open_idx + (min(len(data), self.ticks_per_trading_day) - 1)
        if False:
            start_idx = open_idx + int(np.random.uniform(low=0, high=(self.ticks_per_trading_day - 1)))
        else:
            start_idx = open_idx

        self.current_chart[ic] = dict()
        self.current_chart[ic]['columns'] = columns
        self.tick_to_close = close_idx - start_idx
        self.current_chart[ic]['current_idx'] = start_idx
        self.current_chart[ic]['data'] = np.concatenate([past_data, data], axis=0)  # should be np.ndarray of ohlc needed to simulate a trading day
        self.current_price = self.current_chart[ic]['data'][start_idx][columns.index('close')]

    def _get_num_contract(self):
        if self.position > 0:
            return self.current_account_valuation * self.position / self.current_price
        else:
            return self.current_account_valuation * self.position / (2 * self.average_position_price - self.current_price)

    def get_daily_profit(self):
        return self.current_account_valuation / self.initial_account_valuation - 1

    def print_profit(self):
        if self.tick_to_close == 0:
            print(f'Market closed. Current profit: {self.get_daily_profit() * 100}%')

    def reset(self, code: Optional[str] = None, date_idx: Optional[int] = None):
        # choose code from db and date between [self.start_date, self.end_date]
        if self.db_cursor is None:
            self.db_cursor = self.db_connection.cursor()

        _past_code = self.current_code
        self.current_code = np.random.choice(list(self.validated_code_dict)) if code is None else code
        assert self.current_code in self.validated_code_dict, 'Given code is not in database'

        if _past_code != self.current_code:
            self._parse_date_from_1min_candle()

        self.current_date_idx = np.random.choice(self.simulatable_date_idx_list) if date_idx is None else date_idx
        self.current_date = self.parsed_date_list[self.current_date_idx]

        self.current_chart = dict()
        assert '1분' in self.input_config, '1분봉은 필수'

        for ic in self.input_config:
            if ic == '1분':
                self._prepare_1min_chart()
            elif ic == '60분':
                required_past_dates = math.ceil(self.input_config[ic]['N'] / 7)
                raise NotImplementedError
            elif ic == '1일':
                required_past_dates = math.ceil(self.input_config[ic]['N'])
                raise NotImplementedError

        if self.tick_to_close == self.ticks_per_trading_day - 1:  # cash 100%
            self.average_position_price = 0
            self.position = 0
            self.current_cash = int(np.random.uniform(500, 10000)) * 1000
            self.checkpoint_account_valuation = self.initial_account_valuation = self.current_account_valuation
            self.current_cash = self.initial_account_valuation
        else:
            self.average_position_price = self.current_price + int(np.random.uniform(-30, 30)) * int(self.current_price * 0.001)
            self.position = int(np.random.uniform(-1, 1) * (10000000 // self.current_price))
            self.current_cash = int(np.random.uniform(1, 10000)) * 1000
            self.checkpoint_account_valuation = self.initial_account_valuation = self.current_account_valuation

        return self._observe()

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

        if self.state is None:
            return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


if __name__ == '__main__':
    env = FutureTradingEnv(db_file_path='C:\\Users\\hanseungho\\stocks\\Kiwoom_datareader\\db\\200810_025124_주식분봉_122630.db', start_date=20190801, end_date=20200131, input_config={'1분': {'N': 128, 'columns': ['open', 'high', 'low', 'close']}})

    d = env.reset()
    print(d)
