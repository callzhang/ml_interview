from gsheetsdb import connect
import requests, json, logging
import streamlit as st
import pandas as pd
from streamlit import caching

sheet_url = st.secrets["public_gsheets_url"]
FEISHU_ROBOT_URL = 'https://open.feishu.cn/open-apis/bot/v2/hook/080fd224-7e32-4b3a-ab64-4971df0c3bd1'
ERROR_ROBOT_URL = 'https://open.feishu.cn/open-apis/bot/v2/hook/4c006db0-21fa-4853-b7fa-14bc3b65f94d'
CHAT_ID = 'oc_4fffe5fcd31d362acfd394525ce37118'

def send_message(message: str, type = None):
    data = json.dumps({
        "msg_type": "text",
        "content": {
            "text": message
        }
    })
    if type == 'error':
        res = requests.post(ERROR_ROBOT_URL, data=data)
    else:
        res = requests.post(FEISHU_ROBOT_URL, data=data)

    if res.json().get('code', 0) != 0 and type != 'error':
        msg = res.json().get('msg', '')
        print(f'发送信息失败了：{msg}\n{res.text}\n({message})'[:1000])
        send_message(f'发送消息失败：\n{message}', type='error')
        return False

    print('-'*50 + 'message start' + '-'*50)
    print(message)
    print('-'*50 + 'message end' + '-'*50)
    return True


def upload_record(name:str, record: str):
    url = 'https://feishu-robot-automatnservice-agwxaiqmvf.cn-beijing.fcapp.run/new_record_notification'
    payload = {
        'name': name,
        'record': record
    }
    res = requests.post(url, json=payload)
    assert res.status_code == 200, f'{res}, {res.text}'
    file_key = res.json()
    return file_key


# Create a connection object.
conn = connect()

# Perform SQL query on the Google Sheet.
# Uses st.cache to only rerun when the query changes or after 10 min.
@st.cache(ttl=600)
def get_db():
    query = f'SELECT * FROM "{sheet_url}"'
    rows = conn.execute(query, headers=1)
    rows = rows.fetchall()
    df = pd.DataFrame(rows)
    st.session_state['db'] = df
    logging.info(f'Updated db: \n{df}')
    return df
