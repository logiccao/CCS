
# -*- coding: utf-8 -*-
"""
文件名: chat_api_wrap.py
创建时间: 2025/09/26
作者: logiccao
"""
from flask import Flask, request, jsonify, Response, stream_with_context, g
from flask_cors import CORS
from datetime import datetime
import uuid
import json 
import re 
from chat_llm.check_header import check_header
from chat_llm.logger import setup_logger
logger = setup_logger('CCS', log_file='logs/CCS.log')
app = Flask(__name__)
CORS(app, resources=r'/*')
CORS(app, supports_credentials=True) 
from chat_llm.naive_chat import NativeChat

NativeChator_med_audio = NativeChat(name='naive_med_audio', logger=logger, use_model='deepseek-v3')
check_header(app)

session_cache = {}

def generate_session_id():
    """生成唯一session_id (UUID4 + 时间戳)"""
    return f"sid-{datetime.now().strftime('%Y%m%d%H%M')}-{uuid.uuid4().hex[:8]}"

def check_pattern(text, patterns):
    """检查文本是否包含特定模式"""
    for pattern in patterns:
        if pattern in text:
            return 'true'
    return 'false'

end_pattern = re.compile(r'(?:^|[\s,，.。;；])(?:(?:好的|行|明白了|知道了|没问题|ok)\W*(?:谢谢|感谢|thx|3q)\W*(?:您|你)?\W*(?:再见|拜拜|结束)?[!！。.？?]*$|(?:那就这样|那就到这里|没有(?:其他)?问题了?|我(?:的)?问题(?:解决)?了|不需要了?|可以了?|没事了?|就这样吧)[!！。.？?]*$|(?:谢谢|感谢|多谢|thx|3q)\W*(?:您|你)?\W*(?:啊|啦|呢)?[!！。.？?]*$|(?:再见|拜拜|结束|挂了吧?|停吧?|bye)[\s\W]*$|(?:thank\s*you|thanks|bye|byebye)[\s\W]*$)', 
                         re.IGNORECASE)

def should_end_call(user_input):
    return bool(end_pattern.search(user_input))


@app.route('/chat_audio/naive_med', methods=['POST'])
def naive_med_chat_api():
    """流式对话API接口, triage chat"""
    request_id = g.request_id
    logger.debug(f'request_id :{request_id}')
    success_resp = None
    code = 500
    msg = ''
    try:
        request_data = request.get_json()
        query = request_data['query']
        session_id = request_data.get('session_id', '')
        if session_id == '':
            session_id = generate_session_id()
            session_cache[session_id] = ''
            msg = '新会话'
            logger.info('新会话')
        else:
            if session_id not in session_cache:
                code = 400
                msg = '请求错误：session_id'
            elif session_cache.get(session_id) == 'done':
                code = 400
                msg = '请求错误：会话已结束'
        if code == 400:
            logger.error(f'{msg}, {code}')
            return jsonify({'msg': msg}), code 

        logger.info(f'request data : {request_data}')
        stream_resp = NativeChator_med_audio.chat_with_query(session_id = session_id, query = query) # a generator 
        full_text = ''
        session_finish = 'false'
        def wrap_stream_resp(stream_resp):
            nonlocal full_text
            nonlocal request_id
            nonlocal session_finish
            for chunk in stream_resp:  # 按块读取数据
                if chunk:
                    for choice in chunk.choices:
                        content = choice.delta.content
                        if content:
                            # print(content, end="")
                            full_text += content  # 将每块数据拼接到长文本中
                            response_data = {
                                "text_chunk": content,
                                "session_finish": session_finish,
                                "session_id" : session_id
                            }
                            yield f'id: {request_id}\nevent: message\ndata: {json.dumps(response_data)}\n\n'  # 将数据块返回给客户端
            if len(full_text) > 0:
                end_call = should_end_call(query)
                session_finish = 'true' if end_call else 'false'
                if session_finish == 'true':
                    session_cache[session_id] = 'done'
                response_data = {
                    "text_chunk": '',
                    "session_finish": session_finish,
                    "session_id" : session_id
                }
                yield f'id: {request_id}\nevent: done\ndata: {json.dumps(response_data)}\n\n'

        success_resp = Response(
                wrap_stream_resp(stream_resp = stream_resp), 
                mimetype = "text/event-stream", # 传过来，其实是"text/event-stream"
                status = 200)  
    except Exception as e:
        logger.error(f'chat_api内部出错：{str(e)}')
        code = 500
        return jsonify({'msg' : f'{str(e)}'}), code
    if success_resp is not None:
        @success_resp.call_on_close
        def record_full_text():
            logger.debug(f'request_id :{request_id}')
            NativeChator_med_audio.store_to_history(session_id, full_text)
            logger.debug(f'模型全部回答：{full_text}')
        return success_resp


@app.route('/session/history', methods=['POST'])
def session_history():
    msg = ''
    history = []
    try:
        request_data = request.get_json()
        session_id = request_data['session_id']
        if session_id not in RAGChator.conversations:
            msg = f'{session_id} not in conversations'
        else:
            if 'user_assistant_history' not in RAGChator.conversations[session_id]:
                msg = f'{session_id} has no user_assistant_history'
        if msg == '':
            history = RAGChator.conversions[session_id]['user_assistant_history'] 
            msg = 'success'
        return jsonify({'msg' : msg, 'history': history}), 200
    except Exception as e:
        logger.error(f'api internal error : {str(e)}')


@app.route('/feedback', methods=['POST'])
def feedback_api():
    """用户反馈API接口"""
    request_id = g.request_id
    logger.debug(f'feedback request_id: {request_id}')
    code = 200
    msg = 'success'

    try:
        request_data = request.get_json()

        # 验证必要字段
        required_fields = ['sessionId', 'userQuery', 'assistantResponse']
        for field in required_fields:
            if field not in request_data:
                code = 400
                msg = f'缺少必要字段: {field}'
                logger.error(f'feedback API error: {msg}')
                return jsonify({'msg': msg}), code

        # 提取反馈数据
        session_id = request_data['sessionId']
        user_query = request_data['userQuery']
        assistant_response = request_data['assistantResponse']
        selected_feedback = request_data.get('selectedFeedback', '')
        custom_feedback = request_data.get('customFeedback', '')
        timestamp = request_data.get('timestamp', datetime.now().isoformat())

        # 构建反馈记录
        feedback_record = {
            'feedback_id': f"fb-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}",
            'session_id': session_id,
            'user_query': user_query,
            'assistant_response': assistant_response,
            'selected_feedback': selected_feedback,
            'custom_feedback': custom_feedback,
            'timestamp': timestamp,
            'request_id': request_id
        }

        # 记录到日志
        logger.info(f'收到用户反馈: session_id={session_id}, selected_feedback={selected_feedback}')
        logger.debug(f'反馈详情: {json.dumps(feedback_record, ensure_ascii=False, indent=2)}')
        logger.info(NativeChator_med_audio.system_prompt)

        # 可选：保存到文件或数据库
        # 这里先简单记录到日志，您可以根据需要扩展存储逻辑
        try:
            # 保存到反馈日志文件
            feedback_logger = setup_logger('FEEDBACK', log_file='logs/feedback.log')
            feedback_logger.info(json.dumps(feedback_record, ensure_ascii=False))
        except Exception as log_error:
            logger.warning(f'保存反馈到日志文件失败: {str(log_error)}')

        # 可选：根据反馈类型进行特殊处理
        if selected_feedback in ['inaccurate', 'incomplete']:
            logger.warning(f'收到负面反馈 - session_id: {session_id}, type: {selected_feedback}')
            # 这里可以添加告警或特殊处理逻辑

        msg = '反馈提交成功'

    except KeyError as e:
        code = 400
        msg = f'请求数据格式错误: {str(e)}'
        logger.error(f'feedback API KeyError: {msg}')
    except Exception as e:
        code = 500
        msg = f'服务器内部错误: {str(e)}'
        logger.error(f'feedback API内部错误: {str(e)}')

    return jsonify({
        'msg': msg,
        'code': code,
        'timestamp': datetime.now().isoformat()
    }), code


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5019, debug=False)
    

# nohup python3 -m chat_llm.chat_api_wrap &
