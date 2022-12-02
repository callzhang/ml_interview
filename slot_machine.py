import streamlit as st
import pandas as pd
import numpy as np
import sklearn
# import plotly.figure_factory as ff
import plotly.express as px
from random import sample

st.title('Casino game')
st.subheader('问题')
st.info('''假定你走进一家赌场，你的目标是赢得尽量多的金币。前提：
  1. 赌场里面有很多老虎机（可以考虑为无数台），
  2. 每台收益随机（不同的老虎机之间吐出的金币数量随机）
  3. 对于任意一台老虎机，其每次收益一样（吐出的金币数量一致）
  4. 现在你有100次机会去玩老虎机
  请设计一个策略，这个策略可以让你的期望收益最大。''')



class Casino:
    def __init__(self, n):
        self.n = n
        max_reward = np.random.randint(100)
        self.population = np.random.poisson(max_reward, n)
        self.current = 0
        self.total = 0

    def play(self):
        if self.current < self.n:
            reward = self.population[self.current]
            self.current += 1
            return reward
        else:
            raise Exception('No more chance!')

    def reset(self):
        self.current = 0

    def sample(self, n):
        return sample(self.population.tolist(), n)


TOTAL_PlAY = 100
casino = Casino(TOTAL_PlAY)
st.text(f'前10次样本：{casino.sample(10)} ...')

st.subheader(f'样例代码')
input_str = '''
class RandomPlay:
    def __init__(self, casino, total_play):
        # 传入的casino是整个赌场的类
        self.casino = casino
        self.total_play = total_play
        self.total_reward = 0
        self.observed = []

    def play(self):
        epsilon = 0.2
        for i in range(self.total_play):
            r = np.random.random()
            if r < epsilon or i == 0:
                # 探索一个新的老虎机
                reward = self.casino.play() # play一次新的老虎机
                self.total_reward += reward
                self.observed.append(reward)
            else:
                # 使用回报最大的老虎机
                max_ = max(self.observed)
                self.total_reward += max_
        return self.total_reward # 返回结果
'''
st.code(input_str, language='python')
RandomPlay = None
exec(input_str)

# 自定义代码
input_str = '''
class MyPlay: # 这个类名请不要修改
    def __init__(self, casino, total_play):
        self.casino = casino
        self.total_play = total_play
        self.total_reward = 0
        self.observed = []

    def play(self):
        epsilon = 0.2
        for i in range(self.total_play):
            r = np.random.random()
            if r < epsilon or i == 0:
                reward = self.casino.play()
                self.total_reward += reward
                self.observed.append(reward)
            else:
                max_ = max(self.observed) if self.observed else 0
                self.total_reward += max_
        return self.total_reward
'''

st.subheader('答题区域')
st.info(f'请在下方输入代码，请注意类名为`MyPlay`不要修改，其他参考样例代码')
my_code = st.text_area('请输入代码', input_str, height=400)
casino.reset()
MyPlay = None
exec(my_code)

# 最优策略
from scipy.optimize import curve_fit
from scipy.stats import poisson

class BestPlay:
    def __init__(self, casino, total_play) -> None:
        self.casino = casino
        self.total_play = total_play
        self.total_reward = 0
        self.observed = []

    def play(self):
        free_play_times = 10
        max_ = 0
        # 先玩10次
        for i in range(free_play_times):
            reward = self.casino.play()
            self.observed.append(reward)
            max_ = max(self.observed)
        
        # 开始计算
        # fit with curve_fit
        # parameters, cov_matrix = curve_fit(poisson.pmf, bin_middles, entries)

        

if st.button('执行代码'):

    play_random = RandomPlay(casino, TOTAL_PlAY)
    reward1 = play_random.play()
    st.text(f'使用随机策略回报: {reward1}')
    play2 = MyPlay(casino, TOTAL_PlAY)
    reward2 = play2.play()
    st.text(f'我的策略的回报: {reward2}')
