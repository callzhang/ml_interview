import requests, json

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
    url = 'http://localhost:9000/new_record_notification'
    payload = {
        'name': name,
        'record': record
    }
    res = requests.post(url, json=payload)
    assert res.status_code == 200, res.text
    file_key = res.json()
    return file_key


def new_record_message(file_key):
    url = "https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=user_id"
    content = {
        "title": "新的提交记录",
      		"content": [
                [
                    {
                        "tag": "text",
                        "text": "得到一个新的算法提交记录"
                    }
                ],
                [{
                    "tag": "media",
                    "file_key": {file_key},
                    # "image_key": "img_7ea74629-9191-4176-998c-2e603c9c5e8g"
            }]
		]
    }
    payload = json.dump({
        'receive_id_type': 'chat_id',
        "receive_id": "ou_7d8a6e6df7621556ce0d21922b676706ccs",
        "msg_type": "post",
        "content": content.encode(),
    })
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer {}'
    }
