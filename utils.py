from gsheetsdb import connect
import requests, json, logging
import streamlit as st
import pandas as pd

sheet_url = st.secrets["public_gsheets_url"]
headers = {'Content-Type': 'application/json;charset=utf-8'}
def send_message(message: str, type = None):
    data = json.dumps({
        "msgtype": "text",
        "text": {
            "content": message
        }
    })
    if type == 'error':
        res = requests.post(st.secrets['DINGTALK_ROBOT_URL'], data=data, headers=headers)
    else:
        res = requests.post(st.secrets['DINGTALK_ROBOT_URL'], data=data, headers=headers)

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
    payload = {
        'name': name,
        'record': record
    }
    data = json.dumps({
        "msgtype": "text",
        "text": {
            "content": record
        }
    })
    res = requests.post(st.secrets['DINGTALK_ROBOT_URL'], data=data, headers=headers)
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
