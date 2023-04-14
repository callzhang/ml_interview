### 一个算法工程师的在线测试题

这个任务是受到强化学习的启发，但是我在设计这个任务的时候简化了任务要求，使得即使是没有强化学习基础的同学，也能通过基础的概率统计知识，完成这个任务。这使得这个任务可高可低，宽进严出。

任务：一个关于赌场老虎机策略的代码题

考试目的：
1. 基础概率统计
2. 算法思维：将实际任务通过数学表达
3. 编程能力：转化为可执行的代码

环境变量设置：
创建`.streamlit/secrets.toml`文件，内容如下：
```
public_gsheets_url = "xxx" # 公开的google sheet地址, 用于存储用户信息，表头为：姓名、访问码、截止时间
FEISHU_ROBOT_URL = 'xxx' # 飞书机器人地址（可选）
ERROR_ROBOT_URL = 'xxx' # 报错群聊机器人地址
DINGTALK_ROBOT_URL= 'xxx' # 飞书群聊机器人地址
```

启动方式：
```bash
streamlit run slot_machine.py
```
