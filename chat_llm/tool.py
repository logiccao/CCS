from flask import Flask, request, jsonify, Response, stream_with_context
import uuid
import json
import time
import requests
from threading import Lock
from collections import defaultdict


triage_api_url = "http://127.0.0.1:5019/chat_audio/naive_med"

prompt1 = """1+1="""


def dialogue():
    # 获取会话信息
    request_id = "1234567890"
    user_utterance = prompt1

    # 初始化变量
    buffer = b""  # 初始化为空的字节串
    result = ""   # 初始化为空字符串

    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "Authorization": "BB0015"
    }
    payload = {
        "age": 25,
        "sex": '男',
        "query": user_utterance,
        "session_id": ""
    }

    try:
        triage_response = requests.post(
            url=triage_api_url,
            json=payload,
            headers=headers,
            stream=True,
            timeout=30
        )
        print(triage_response.status_code)

        for chunk in triage_response.iter_content(chunk_size=1024):
            if not chunk:
                continue

            buffer += chunk
            events = buffer.split(b"\n\n")
            buffer = events.pop()

            for event in events:
                event = event.decode('utf-8').strip()
                if not event:
                    continue

                lines = event.split('\n')
                event_data = {}
                for line in lines:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        event_data[key.strip()] = value.strip()

                if event_data.get('event') in ['message', 'done']:
                    data = json.loads(event_data['data'])
                    result += data.get('text_chunk', '')
                    print(data)

    except requests.exceptions.RequestException as e:
        print(f"请求错误: {e}")
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
    except Exception as e:
        print(f"其他错误: {e}")

dialogue()