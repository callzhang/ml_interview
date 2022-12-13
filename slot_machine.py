from rich.logging import RichHandler
import streamlit as st
import pandas as pd
import numpy as np
from utils import *
import logging
from rich.traceback import install
from datetime import datetime
install(show_locals=True, max_frames=5, suppress=[st])
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True,
                          tracebacks_show_locals=True, tracebacks_suppress=[st])],
)

st.title('Casino game')
if 'db' not in st.session_state:
    db = get_db()
    st.session_state['db'] = db
key = st.text_input('请输入访问码，没有访问码请找HR询问。')
if key:
    access_data = st.session_state['db'].query('访问码==@key')
    if len(access_data) == 0:
        st.warning('未授权，请重试！')
        st.stop()
    myname = access_data['姓名'].iloc[0]
    st.session_state['myname'] = myname
    expiration = access_data['截止日期'].iloc[0]
    if datetime.now().date() > expiration:
        st.info(f'{myname}，本账户已截止，请联系HR。')
        st.stop()
    else:
        st.info(f'{myname}，欢迎您！您的截止日期为：{expiration}')
else:
    st.stop()
st.subheader('问题')
st.info('''假定你走进一家赌场，我的目标是赢得尽量多的金币。前提：
  1. 赌场里面有很多老虎机（可以考虑为无数台），
  2. 每台收益随机（不同的老虎机之间吐出的金币数量随机）
  3. 对于任意一台老虎机，其每次收益一样（单个老虎机的回报固定）
  4. 现在你有n次机会去玩老虎机(n>100)
  请设计一个策略，这个策略可以让我的期望收益最大。''')


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
            raise Exception('你已超过尝试次数!')

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
        self.total_reward = 0
        # 其他初始化代码

    def play(self):
        # 填入你的策略
        return self.total_reward
        
'''

st.subheader('答题区域')
st.info(f'请在下方输入代码，请注意类名为`MyPlay`不要修改，其他参考样例代码')
my_code = st.text_area('请输入代码', my_code, height=300)
casino.reset()
MyPlay = None
exec(my_code)

# 基准策略
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
        
score_result = {
    '远低于平均的小白': range(int(-1e5),92),
    '新手请继续努力': range(92, 95),
    '有点感觉的小学生': range(95, 98),
    '有经验的玩家': range(98,100),
    '赌场一霸': range(100, int(1e5)),
}

if not myname:
    st.warning('请填写名字')
if st.button('执行我的策略') and myname:
    # 进度
    bar = st.progress(0)
    ph = st.empty()
    header = ['我的策略', '基准策略', '评分']
    result = pd.DataFrame(columns=header)
    ph.table(result)
    score_best = 0
    sample_best = None
    stable = True
    for i in range(100):
        bar.progress(i+1)
        casino = Casino(TOTAL_PlAY + i*10)
        play_random = RandomPlay(casino, TOTAL_PlAY + i*10)
        play2 = MyPlay(casino, TOTAL_PlAY + i*10)
        bestplay = BestPlay(casino, TOTAL_PlAY + i*10)
        # 测试随机策略
        reward_random = play_random.play()
        print(f'使用随机策略回报: {reward_random}')
        # st.text(f'使用随机策略回报: {reward1}')
        # 测试我的策略
        casino.reset()
        my_reward = play2.play()
        print(f'我的策略的回报: {my_reward}')
        # st.text(f'我的策略的回报: {reward2}')
        # 测试最佳策略
        casino.reset()
        reward_benchmark = bestplay.play()
        print(f'最佳回报：{reward_benchmark}')
        # st.text(f'最佳回报：{reward3}')
        score = int((my_reward - reward_random) / (reward_benchmark - reward_random) *100) if reward_benchmark > reward_random and reward_benchmark > my_reward else int(my_reward/reward_benchmark*100)
        if score > score_best:
            sample_best = casino.sample
        rewards = pd.Series([my_reward, reward_benchmark, score], index=header)
        result = result.append(rewards, ignore_index=True)
        ph.table(result)
        average = result.mean()
        avg_score = average['评分']
        # 测试一下结果稳定性
        if i >= 9:
            dist = distfit(distr=['norm'], smooth=10)
            dist.fit_transform(np.array(result['评分']), verbose=1)
            mean = dist.model['params'][0]
            scale = dist.model['params'][1]
            print(dist.model)
            if scale/mean > 0.5:
                st.warning('结果不稳定，请优化算法')
                stable = False
                break
        if avg_score < 98 and i >= 9:
            break
    bar.progress(100)
    for k, v in score_result.items():
        if int(avg_score) in v:
            comment = k
            break
    st.info(f'你的成绩是：{comment}')
    st.session_state['comment'] = comment
    # 生成记录
    record = f'''
# 新的算法提交（{myname}）
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
    st.session_state['record'] = record
    st.session_state['score'] = avg_score

    # 记录
    if average['评分'] > 100 and stable:
        st.balloons()
        st.text(f'老虎机的回报：{casino.sample} ...')
        with open('sample.md', 'a') as f:
            f.write(record)
            print(record)
        # 下载记录
        st.success('恭喜你打败基准策略')
        st.download_button('下载记录', record, file_name='测试记录.md')


if st.button('提交策略'):
    if 'record' not in st.session_state:
        st.error('请先执行策略')
    elif st.session_state['score'] < 96:
        st.warning('请先优化策略再提交')
        upload_str = f"{st.session_state['myname']}尝试提交，但是成绩不够好，成绩为{st.session_state['score']}({st.session_state['comment']})"
        result = upload_record(st.session_state['myname'], upload_str)
    else:
        logging.info('提交结果中')
        result = upload_record(st.session_state['myname'], st.session_state['record'])
        st.info(result)
