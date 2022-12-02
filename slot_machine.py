import streamlit as st
import pandas as pd
import numpy as np

st.title('Casino game')
st.subheader('问题')
st.info('''假定你走进一家赌场，你的目标是赢得尽量多的金币。前提：
  1. 赌场里面有很多老虎机（可以考虑为无数台），
  2. 每台收益随机（不同的老虎机之间吐出的金币数量随机）
  3. 对于任意一台老虎机，其每次收益一样（单个老虎机的回报固定）
  4. 现在你有n次机会去玩老虎机(n>100)
  请设计一个策略，这个策略可以让你的期望收益最大。''')



class Casino:
    def __init__(self, n):
        self.n = n
        max_reward = np.random.randint(5, 20)
        population = np.random.poisson(max_reward, n) 
        population -= int(max_reward / 2)
        population[population < 0] = 0
        self.population = population
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

    def get_sample(self):
        return self.population.tolist()
    sample = property(get_sample)

TOTAL_PlAY = 100
casino = Casino(TOTAL_PlAY)

st.subheader(f'样例代码')
sample_code = '''
import scipy, statsmodels, distfit, sklearn, pandas as pd, numpy as np
class RandomPlay:
    def __init__(self, casino, total_play):
        # casino：赌场实例，其play()函数为探索一台新的老虎机
        # total_play：总共可以尝试的次数
        self.casino = casino
        self.total_play = total_play
        self.total_reward = 0
        self.observed = []

    def play(self): # 系统会调用这个函数执行策略
        epsilon = 0.5
        for i in range(self.total_play):
            r = np.random.random()
            if r < epsilon or i == 0:
                # 探索一个新的老虎机，调用赌场的play()函数
                reward = self.casino.play()
                self.total_reward += reward
                self.observed.append(reward)
            else:
                # 使用回报最大的老虎机
                max_ = max(self.observed)
                self.total_reward += max_
        return self.total_reward # 返回结果
'''
st.code(sample_code, language='python')
RandomPlay = None
exec(sample_code)

# 自定义代码
my_code = '''
class MyPlay: # 这个类名请不要修改
    def __init__(self, casino, total_play):
        self.casino = casino
        self.total_play = total_play
        self.total_reward = 0
        self.observed = []

    def play(self): # 这个函数是必须的
        for i in range(self.total_play):
            r = np.random.random()
            reward = self.casino.play()
            self.total_reward += reward
            self.observed.append(reward)
        return self.total_reward
'''

st.subheader('答题区域')
st.info(f'请在下方输入代码，请注意类名为`MyPlay`不要修改，其他参考样例代码')
my_code = st.text_area('请输入代码', my_code, height=300)
casino.reset()
MyPlay = None
exec(my_code)

# 最优策略
from distfit import distfit
class BestPlay:
    def __init__(self, casino, total_play) -> None:
        self.casino = casino
        self.total_play = total_play
        self.total_reward = 0
        self.observed = []
        self.played = 0

    def fit_dist(self):
        dist = distfit(distr=['gamma', 'lognorm', 'beta', 't', 'chi'], smooth=10)
        dist.fit_transform(np.array(self.observed), verbose=1)
        score = dist.model['score'] / len(self.observed)
        return dist, score

    def play(self):
        free_play_times = 10
        # 先玩10次
        for i in range(self.total_play):
            if i < free_play_times:
                self.explore()
                continue
            
            # calculate gains and loss, skip if starting to exploit
            if self.played - len(self.observed) < 3:
                # fit with curve_fit
                dist, score = self.fit_dist()

                if score > 1:  # if RSS is too large, continue explore
                    self.explore()
                    continue
                y = range(self.max*5)
                # proba = dist.predict(y)['y_proba']
                p_value = dist.model['model'].cdf(y)
                proba = p_value
                proba[1:] = p_value[1:]-p_value[:-1]
                loss = sum([(self.max-y_)*p for y_, p in zip(y, proba) if y_<self.max])
                steps_left = (self.total_play-self.played)
                gain = sum([(y_-self.max)*p for y_,p in zip(y, proba) if y_>self.max]) * steps_left
                if loss > gain and gain>0:
                    self.exploit()
                else:
                    self.explore()
                    print(f'Step {self.played}: loss[{loss}] | gain[{gain}] | total: {self.total_reward}')
            else:
                self.exploit()
        return self.total_reward
            

    def explore(self):
        reward = self.casino.play()
        self.observed.append(reward)
        self.played += 1
        self.total_reward += reward
        print(f'Step {self.played}: explore, reward: {reward}')
        return reward

    def exploit(self):
        self.played += 1
        self.total_reward += self.max
        return self.max
    
    def get_max(self):
        return max(self.observed)

    max = property(get_max)
        

if st.button('执行我的策略'):
    bar = st.progress(0)
    ph = st.empty()
    header = ['随机策略', '我的策略', '最优答案', '得分']
    result = pd.DataFrame(columns=header)
    ph.table(result)
    score_best = 0
    sample_best = None
    for i in range(100):
        bar.progress(i+1)
        casino = Casino(TOTAL_PlAY + i*10)
        play_random = RandomPlay(casino, TOTAL_PlAY + i*10)
        play2 = MyPlay(casino, TOTAL_PlAY + i*10)
        bestplay = BestPlay(casino, TOTAL_PlAY + i*10)
        # 测试随机策略
        reward1 = play_random.play()
        print(f'使用随机策略回报: {reward1}')
        # st.text(f'使用随机策略回报: {reward1}')
        # 测试我的策略
        casino.reset()
        reward2 = play2.play()
        print(f'我的策略的回报: {reward2}')
        # st.text(f'我的策略的回报: {reward2}')
        # 测试最佳结果
        casino.reset()
        reward3 = bestplay.play()
        print(f'最佳回报：{reward3}')
        # st.text(f'最佳回报：{reward3}')
        score = int(reward2/reward3*100)
        if score > score_best:
            sample_best = casino.sample
        print(f'您的算法得分：{int(reward2/reward3*100)}')
        rewards = pd.Series([reward1, reward2, reward3, score], index=header)
        result = result.append(rewards, ignore_index=True)
        ph.table(result)
        average = result.mean()
        if average['得分'] < 98 and i >= 9:
            break
    bar.progress(100)
    
    st.info('平均得分')
    st.table(average)
    record = f'''
# New record
##老虎机序列：
```
{str(sample_best)}
```
## 游戏记录
{result.to_markdown()}
## 平均成绩
{average.to_markdown()}
## 策略代码
```python
{my_code}
```
'''+'-'*100

    # 记录
    if average['我的策略'] > average['最优答案']:
        st.balloons()
        st.text(f'老虎机的回报：{casino.sample} ...')
        with open('sample.md', 'a') as f:
            f.write(record)
            print(record)
        # 下载记录
        st.success('恭喜你打败最优答案')
        st.download_button('下载记录', record, file_name='成绩单.md')
