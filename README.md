### ML Engineer Online Test
[中文](README_CN.md)

A server for online tests for ML Engineer candidates.

> This task is inspired by reinforcement learning, but I simplified the task requirements when designing it, so that even students without a basic understanding of Reinforcement Learning can complete the task through basic probability and statistics. This makes the task easier to be solved but hard to be perfect, depending on the examiner's own choice.

**Task:**
A coding task to design a strategy to play casino slot machines.

**Test objectives:**
- Basic probability and statistics
- Algorithm thinking: mathematically express practical tasks
- Programming ability: convert to executable code.

**Environment variable set up:**
Create a.streamlit/secrets.tomlfile with the following contents:
```
public_gsheets_url = "xxx" # Public Google Sheet address for storing user information, with headers: name, access code, deadline
FEISHU_ROBOT_URL = 'xxx' # Feishu robot address (optional)
ERROR_ROBOT_URL = 'xxx' # Error group chat robot address
DINGTALK_ROBOT_URL= 'xxx' # DingTalk group chat robot address
```

**Start up the server**
```bash 
streamlit run slot_machine.py
```
