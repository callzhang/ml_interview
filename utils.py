# Compatibility shim for Python 3.12+ with gsheetsdb dependencies
import collections
import collections.abc
# Patch collections.Iterable for Python 3.12 compatibility
if not hasattr(collections, 'Iterable'):
    collections.Iterable = collections.abc.Iterable

# Import gsheetsdb with error suppression for dependency warnings
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    from gsheetsdb import connect
    
# Additional compatibility shims for mo_parsing issues
try:
    import mo_parsing.debug
    if not hasattr(mo_parsing.debug, 'DEBUGGING'):
        mo_parsing.debug.DEBUGGING = False
except (ImportError, AttributeError):
    pass

# Monkey patch for SQL parsing issues with gsheetsdb
# This needs to happen after gsheetsdb is imported but before it's used
try:
    import gsheetsdb.query
    import moz_sql_parser
    _original_parse = moz_sql_parser.parse
    
    def _patched_parse(sql):
        # Ensure the SQL string is properly handled
        if isinstance(sql, bytes):
            sql = sql.decode('utf-8')
        sql = sql.strip()
        # Try the original parser
        return _original_parse(sql)
    
    # Patch both the module-level function and gsheetsdb's reference
    moz_sql_parser.parse = _patched_parse
    if hasattr(gsheetsdb.query, 'parse_sql'):
        gsheetsdb.query.parse_sql = _patched_parse
except (ImportError, AttributeError):
    pass
import requests, json, logging
import streamlit as st
import pandas as pd

sheet_url = st.secrets["public_gsheets_url"]
dingtalk_url = st.secrets['DINGTALK_ROBOT_URL']
headers = {'Content-Type': 'application/json;charset=utf-8'}
def send_message(message: str, type = None):
    data = json.dumps({
        "msgtype": "text",
        "text": {
            "content": message
        }
    })
    if type == 'error':
        res = requests.post(dingtalk_url, data=data, headers=headers)
    else:
        res = requests.post(dingtalk_url, data=data, headers=headers)

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
    res = requests.post(dingtalk_url, data=data, headers=headers)
    assert res.status_code == 200, f'{res}, {res.text}'
    file_key = res.json()
    return file_key


# Create a connection object.
conn = connect()

# Perform SQL query on the Google Sheet.
# Uses st.cache_data to only rerun when the query changes or after 10 min.
@st.cache_data(ttl=600)
def get_db():
    query = f'SELECT * FROM "{sheet_url}"'
    try:
        rows = conn.execute(query, headers=1)
        rows = rows.fetchall()
        df = pd.DataFrame(rows)
        st.session_state['db'] = df
        logging.info(f'Updated db: \n{df}')
        return df
    except Exception as e:
        # Enhanced error handling for SQL parsing issues
        error_msg = str(e)
        if 'ParseException' in error_msg or 'Expecting ordered sql' in error_msg:
            logging.error(f'SQL parsing error with query: {query}')
            logging.error(f'Error details: {error_msg}')
            # Try alternative query format
            try:
                # Some versions may need different quoting
                alt_query = f"SELECT * FROM {sheet_url}"
                rows = conn.execute(alt_query, headers=1)
                rows = rows.fetchall()
                df = pd.DataFrame(rows)
                st.session_state['db'] = df
                logging.info(f'Updated db with alternative query format: \n{df}')
                return df
            except Exception as e2:
                logging.error(f'Alternative query also failed: {e2}')
        raise
