from rich.logging import RichHandler
import streamlit as st
import pandas as pd
import numpy as np
from storing import stable_score, check_unstable
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
    elif key == 'stardust2017':
        st.warning(f' **管理员欢迎您！** 您可以访问[表格](https://docs.google.com/spreadsheets/d/1fulK_LYWk2zHMo4qjzdUQNif_2VyzVJFH9Xb-XXzxB0/edit?pli=1&gid=0#gid=0)修改登录，或修改代码[Github](https://github.com/callzhang/ml_interview)')
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
        samples = np.random.poisson(max_reward, n) 
        samples -= int(max_reward / 2)
        samples[samples < 0] = 0
        self.samples = samples
        self.rewards = [] # 记录所有游戏的结果
        
    def play_machine(self, machine_id):
        assert 0 <= machine_id < len(self.samples), f'机器ID {machine_id} 超出范围 [0, {len(self.samples)-1}]'
        assert len(self.rewards) < self.n, '你已超过尝试次数，请检查代码！'
        reward = self.samples[machine_id]
        self.rewards.append(reward)
        return reward
    
    def reset(self):
        self.rewards = []
    
    def get_total_reward(self):
        return sum(self.rewards)

    def get_sample(self):
        return self.samples.tolist()
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

    def play(self): # 系统会调用`total_play`次本函数执行策略，每次需要调用`play_machine(machine_id)`选择机器
        self.played += 1
        # 随机选择一个机器ID (0 到 total_play-1)
        machine_id = np.random.randint(0, self.total_play)
        reward = self.casino.play_machine(machine_id)
        self.observed.append(reward)
        return

'''
st.code(sample_code, language='python')
CasinoPlay = None
exec(sample_code)

# 自定义代码
my_code = '''class MyPlay(CasinoPlay): #这个类名请不要修改，并提供`play`函数供系统调用
    def play(self): # 请注意，这个函数会被系统调用total_play次
        # 填入你的策略，使用 play_machine(machine_id) 选择机器, machine_id=[0, total_play-1] 
        reward = self.casino.play_machine(0) # 选择第0台机器, 你也可以选择其他机器
        self.observed.append(reward) # 记录收益
        return
    
    def my_function(self): # 你可以自定义一些函数，用于中间步骤，不会被系统调用
        print('my_function')
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
        self.machine_rewards = {}  # 记录每个机器的奖励 {machine_id: reward}

    def fit_dist(self):
        dist = distfit(distr=['gamma', 'lognorm', 'beta', 't', 'chi'], smooth=10)
        dist.fit_transform(np.array(self.observed))
        score = dist.model['score'] / len(self.observed)
        return dist, score

    def play(self):
        self.played += 1
        # 先探索 free_play_times 次
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
        # 探索：随机选择一个未尝试过的机器
        available_machines = list(range(self.total_play))
        untried = [m for m in available_machines if m not in self.machine_rewards]
        if untried:
            machine_id = np.random.choice(untried)
        else:
            # 如果所有机器都尝试过，随机选择一个
            machine_id = np.random.choice(available_machines)
        
        reward = self.casino.play_machine(machine_id)
        self.machine_rewards[machine_id] = reward
        self.observed.append(reward)
        self.total_reward += reward
        # print(f'Step {self.played}: explore, reward: {reward}')
        return reward

    def exploit(self):
        # 利用：选择已知最好的机器
        best_machine_id = max(self.machine_rewards, key=self.machine_rewards.get)
        reward = self.casino.play_machine(best_machine_id)
        self.total_reward += reward
        assert reward == self.max, 'exploit error!'
        return reward
    
    def get_max(self):
        return max(self.observed) if self.observed else 0

    max = property(get_max)
        
score_ranking = {
    '远低于平均的小白': (-float('inf'), 0.92),
    '新手请继续努力': (0.92, 0.95),
    '有点感觉了': (0.95, 0.98),
    '有经验的玩家，请继续努力': (0.98, 1.0),
    '赌场一霸': (1.0, float('inf')),
}

is_unstable = False
if st.button('执行我的策略') and myname:
    # 进度
    bar = st.progress(0)
    result_container = st.empty()
    stats_container = st.empty()
    header = ['我的策略', '基准策略', '评分']
    result = pd.DataFrame(columns=header)
    result_container.table(result)
    for i in range(100):
        bar.progress(i+1)
        total_play = TOTAL_PlAY + i*10
        casino = Casino(total_play)
        play_random = CasinoPlay(casino, total_play)
        play_mine = MyPlay(casino, total_play)
        bestplay = BestPlay(casino, total_play)
        # 测试随机策略
        for j in range(total_play):
            play_random.play()
        reward_random = casino.get_total_reward()
        # print(f'使用随机策略回报: {reward_random}')

        # 测试我的策略
        casino.reset()
        for j in range(total_play):
            play_mine.play()
        my_reward = casino.get_total_reward()
        # print(f'我的策略的回报: {my_reward}')
        
        # 测试最佳策略
        casino.reset()
        for j in range(total_play):
            bestplay.play()
        reward_benchmark = casino.get_total_reward()
        # print(f'最佳回报：{reward_benchmark}')
        
        # 评分
        # score = (my_reward - reward_random) / (reward_benchmark - reward_random) if reward_benchmark > reward_random and reward_benchmark > my_reward else my_reward / reward_benchmark
        score = stable_score(my_reward, reward_random, reward_benchmark)
        print(f'{i}: 随机: {reward_random}, 我的策略: {my_reward}, 最佳回报：{reward_benchmark}, 评分：{score:.3f}')
        # 记录
        rewards = pd.Series([my_reward, reward_benchmark, score], index=header)
        result = pd.concat([result, rewards.to_frame().T], ignore_index=True)

        result_container.table(result)
        
        # Calculate and display statistics
        scores = result['评分']
        my_rewards = result['我的策略']
        best_rewards = result['基准策略']
        
        # 测试结果稳定性
        is_unstable = check_unstable(scores)
        if is_unstable:
            st.warning(f'结果不稳定，请优化算法')
            break
    
    avg_score = result.mean()['评分']
    stats = [
        ('Mean (平均)', f"{scores.mean():.4f}", '要求均值大于0.95或std小于0.25'),
        ('Median (中位数)', f"{scores.median():.4f}", '要求中位数大于0.9'),
        ('Std Dev (标准差)', f"{scores.std():.4f}", '要求标准差小于0.5'),
        ('Min (最小值)', f"{scores.min():.4f}", '要求最小值大于0.1'),
        ('Max (最大值)', f"{scores.max():.4f}", ''),
        ('10% (10%分位数)', f"{scores.quantile(0.1):.4f}", '要求10%分位数大于0.5'),
        ('Win Rate (胜率)', f"{(my_rewards > best_rewards).sum() / len(result) * 100:.2f}%", ''),
    ]
    stats_df = pd.DataFrame(stats, columns=['统计指标', '数值', '备注'])
    stats_container.table(stats_df)
    if not is_unstable:
        comment = None
        for k, v in score_ranking.items():
            if v[0] <= avg_score < v[1]:
                comment = k
                break
        if comment is None:
            comment = f'{avg_score:.3f}'
        st.info(f'你的成绩是：{comment}')
        st.session_state['comment'] = comment
    
    

    # 记录
    if avg_score > 1.0 and not is_unstable:
        record = f'''
# 新的算法提交（{myname}）

## 平均成绩
{avg_score:.3f} ({comment})
## 游戏记录
{result.to_markdown() if not is_unstable else stats_df.to_markdown()}
## 策略代码
```python
{my_code}
```
'''+'-'*100
        ## 生成记录
        st.session_state['record'] = record
        st.session_state['score'] = avg_score
        st.balloons()
        print(record)
        # 下载记录
        st.success('恭喜你打败基准策略')
        st.download_button('下载记录', record, file_name='测试记录.md')


if st.button('提交策略', disabled=is_unstable):
    if 'record' not in st.session_state:
        st.error('请先执行策略')
    elif st.session_state.score < 0.95:
        st.warning('请先优化策略再提交')
        upload_str = f"【{st.session_state['myname']}】尝试提交，但是成绩不够好。\n其成绩为{st.session_state['score']}({st.session_state['comment']})"
        result = upload_record(st.session_state['myname'], upload_str)
        print(result)
    else:
        logging.info('提交结果中')
        result = upload_record(st.session_state['myname'], st.session_state['record'])
        st.info(result)
