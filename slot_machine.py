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

TOTAL_PlAY = 100

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
st.info('''假定你走进一家赌场，你的目标是赢得尽量多的金币。前提：
  1. 赌场里面有很多老虎机（可以考虑为无数台），
  2. 每台收益随机（不同的老虎机之间吐出的金币数量随机）
  3. 对于任意一台老虎机，其每次收益一样（单个老虎机的回报固定）
  4. 现在你有n次机会去玩老虎机(n>100)
  请设计一个策略，这个策略可以让期望收益最大。''')


class Casino:
    def __init__(self, n):
        self.n = n
        max_reward = np.random.randint(5, 200)
        population = np.random.poisson(max_reward, n) 
        population -= int(max_reward / 2)
        population[population < 0] = 0
        self.population = population
        self.tried = 0
        self.history = [] # 记录每次探索的结果
        self.rewards = [] # 记录探索+穷尽的结果

    def play_new(self):
        reward = self.population[self.tried]
        self.tried += 1
        self.history.append(reward)
        self.rewards.append(reward)
        assert len(self.rewards) <= self.n, '你已超过尝试次数，请检查代码！'
        return reward
        
    def play_machine(self, machine_id):
        assert machine_id < len(self.history), '指定机器超出已尝试机器范围，请检查代码！'
        reward = self.history[machine_id]
        self.rewards.append(reward)
        assert len(self.rewards) <= self.n, '你已超过尝试次数，请检查代码！'
        return reward
    
    def reset(self):
        self.tried = 0
        self.rewards = []
    
    def get_total_reward(self):
        return sum(self.rewards)

    def get_sample(self):
        return self.population.tolist()
    sample = property(get_sample)


st.subheader(f'样例代码')
sample_code = '''
import scipy, statsmodels, sklearn, pandas as pd, numpy as np
class CasinoPlay:
    def __init__(self, casino, total_play): 
        self.casino = casino # 赌场实例
        self.total_play = total_play # 总共可以尝试的次数
        self.played = 0
        self.observed = []

    def play(self): # 系统会调用`total_play`次本函数执行策略，每次需要决定`play_new`或`play_machine`
        self.played += 1
        if np.random.rand() < 0.5 or not self.observed:
            # 探索一个新的老虎机，调用赌场的play_new()函数，返回一个新的reward
            reward = self.casino.play_new()
            self.observed.append(reward)
        else:
            # 选择过去的老虎机，可以从历史记录中挑一个，指定machine_id
            machine_id = np.random.choice(range(len(self.observed)))
            reward = self.casino.play_machine(machine_id)
            assert reward == self.observed[machine_id]
        return
'''
st.code(sample_code, language='python')
CasinoPlay = None
exec(sample_code)

# 自定义代码
my_code = '''class MyPlay(CasinoPlay):
    def play(self):
        # 请注意，这个函数会被系统调用total_play次
        # 填入你的策略，使用以下两个方法中的一个
        # reward = self.casino.play_new() # 调用这个函数可以探索一个新的老虎机
        # self.casino.play_machine(0) # 调用这个函数可以使用一个已知老虎机
        return
'''

# ''' 随机较优方案
    # def play(self):
    #     self.played += 1
    #     if self.played <= int(self.total_play**0.5):
    #         reward = self.casino.play_new() # 调用这个函数可以探索一个新的老虎机
    #         self.observed.append(reward)
    #     else:
    #         max_machine = np.argmax(self.observed)
    #         self.casino.play_machine(max_machine) # 调用这个函数可以使用一个已知老虎机
    #     return
# '''

st.subheader('答题区域')
my_code = st.text_area('请输入代码', my_code, height=300, help='请注意类名为`MyPlay`不要修改，并提供`play`函数供系统调用')
# if ' st.' in my_code or 'streamlit' in my_code:
#     st.warning('代码中引用了非法库，请删除！')
#     st.stop()
casino = Casino(TOTAL_PlAY)
MyPlay = None
exec(my_code)

# 基准策略
from distfit import distfit
class BestPlay(CasinoPlay):
    def __init__(self, casino, total_play) -> None:
        super().__init__(casino, total_play)
        self.free_play_times = int(total_play**0.5)
        self.total_reward = 0

    def fit_dist(self):
        dist = distfit(distr=['gamma', 'lognorm', 'beta', 't', 'chi'], smooth=10)
        dist.fit_transform(np.array(self.observed), verbose=1)
        score = dist.model['score'] / len(self.observed)
        return dist, score

    def play(self):
        self.played += 1
        # 先玩10次
        if self.played <= self.free_play_times:
            self.explore()
            
        # calculate gains and loss, skip if starting to exploit
        elif self.played - len(self.observed) < 3:
            # fit with curve_fit
            dist, score = self.fit_dist()

            if score > 1:  # if RSS is too large, continue explore
                self.explore()
            else:
                y = range(self.max*5)
                # proba = dist.predict(y)['y_proba']
                p_value = dist.model['model'].cdf(y)
                proba = p_value.copy()
                proba[1:] = p_value[1:]-p_value[:-1]
                loss = sum([(self.max-y_)*p for y_, p in zip(y, proba) if y_<self.max])
                steps_left = (self.total_play-self.played)
                gain = sum([(y_-self.max)*p for y_,p in zip(y, proba) if y_>self.max]) * steps_left
                if loss > gain and gain>0:
                    self.exploit()
                else:
                    self.explore()
                    # print(f'Step {self.played}: loss[{loss}] | gain[{gain}] | total: {self.total_reward}')
        else:
            self.exploit()
        return

    def explore(self):
        reward = self.casino.play_new()
        self.observed.append(reward)
        self.total_reward += reward
        # print(f'Step {self.played}: explore, reward: {reward}')
        return reward

    def exploit(self):
        self.total_reward += self.max
        max_idx = self.observed.index(self.max)
        reward = self.casino.play_machine(max_idx)
        assert reward == self.max, 'exploit error!'
        return self.max
    
    def get_max(self):
        return max(self.observed)

    max = property(get_max)
        
score_result = {
    '远低于平均的小白': range(int(-1e5),92),
    '新手请继续努力': range(92, 95),
    '有点感觉了': range(95, 98),
    '有经验的玩家': range(98,100),
    '赌场一霸': range(100, int(1e5)),
}

stable = None
if st.button('执行我的策略') and myname:
    # 进度
    bar = st.progress(0)
    ph = st.empty()
    header = ['我的策略', '基准策略', '评分']
    result = pd.DataFrame(columns=header)
    ph.table(result)
    score_best = 0
    for i in range(100):
        bar.progress(i+1)
        total_play = TOTAL_PlAY + i*10
        casino = Casino(total_play)
        play_random = CasinoPlay(casino, total_play)
        play2 = MyPlay(casino, total_play)
        bestplay = BestPlay(casino, total_play)
        # 测试随机策略
        for j in range(total_play):
            play_random.play()
        reward_random = casino.get_total_reward()
        # print(f'使用随机策略回报: {reward_random}')

        # 测试我的策略
        casino.reset()
        for j in range(total_play):
            play2.play()
        my_reward = casino.get_total_reward()
        # print(f'我的策略的回报: {my_reward}')
        
        # 测试最佳策略
        casino.reset()
        for j in range(total_play):
            bestplay.play()
        reward_benchmark = casino.get_total_reward()
        # print(f'最佳回报：{reward_benchmark}')
        
        # 评分
        score = int((my_reward - reward_random) / (reward_benchmark - reward_random) *100) if reward_benchmark > reward_random and reward_benchmark > my_reward else int(my_reward/reward_benchmark*100)
        print(f'{i}: 随机: {reward_random}, 我的策略: {my_reward}, 最佳回报：{reward_benchmark}, 评分：{score}')
        # 记录
        rewards = pd.Series([my_reward, reward_benchmark, score], index=header)
        result = result.append(rewards, ignore_index=True)

        ph.table(result)
        average = result.mean()
        avg_score = average['评分']
        # 测试结果稳定性
        if i >= 9:
            try:
                scores = result['评分']
                dist = distfit(distr=['norm'], smooth=10)
                dist.fit_transform(scores, verbose=1)
                mean = dist.model['params'][0]
                scale = dist.model['params'][1]
                print(dist.model)
            except:
                scale, mean = 0, scores.mean()
                
            if scale/mean > 0.5 or (scores.min()<90 and score.mean()>95):
                st.warning(f'结果不稳定，请优化算法')
                print(f'scale/mean: {scale/mean}, min: {scores.min()}, mean: {scores.mean()}')
                break
        if avg_score < 98 and i >= 9:
            break
    bar.progress(100)
    for k, v in score_result.items():
        if int(avg_score) in v:
            comment = k
            break
        comment = f'{avg_score}'
    st.info(f'你的成绩是：{comment}')
    st.session_state['comment'] = comment
    
    
    ## 生成记录
    record = f'''
# 新的算法提交（{myname}）

## 平均成绩
{average.to_markdown()}
## 游戏记录
{result.to_markdown()}
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
        print(record)
        # 下载记录
        st.success('恭喜你打败基准策略')
        st.download_button('下载记录', record, file_name='测试记录.md')


if st.button('提交策略'):
    if 'record' not in st.session_state:
        st.error('请先执行策略')
    elif st.session_state.score < 95:
        st.warning('请先优化策略再提交')
        upload_str = f"【{st.session_state['myname']}】尝试提交，但是成绩不够好。\n其成绩为{st.session_state['score']}({st.session_state['comment']})"
        result = upload_record(st.session_state['myname'], upload_str)
        print(result)
    else:
        logging.info('提交结果中')
        result = upload_record(st.session_state['myname'], st.session_state['record'])
        st.info(result)
