
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
        knowledge = True if "knowledge" == request_data.get("dialog_type") else False
        stream_resp = NativeChator_med_audio.chat_with_query(session_id=session_id, query=query, knowledge=knowledge) # a generator 
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
    """用户反馈API接口 - 标准化版本，所有反馈立即生效"""
    request_id = g.request_id
    logger.debug(f'feedback request_id: {request_id}')
    code = 200
    msg = 'success'
    optimization_type = None  # 记录优化类型

    # try:
    request_data = request.get_json()

    # 验证必要字段
    required_fields = ['sessionId', 'userQuery', 'assistantResponse', 'dialogType', 'problemSolved', 'rating', 'timestamp']
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
    dialogType = request_data['dialogType']
    problemSolved = request_data['problemSolved']
    rating = request_data['rating']
    timestamp = request_data.get('timestamp', datetime.now().isoformat())

    logger.info(f"session_id: {session_id}")
    logger.info(f"user_query: {user_query}")
    logger.info(f"assistant_response: {assistant_response}")
    logger.info(f"dialogType: {dialogType}")
    logger.info(f"problemSolved: {problemSolved}")
    logger.info(f"rating: {rating}")


    # # 映射前端反馈类型到后端
    # feedback_type_mapping = {
    #     'helpful': 'helpful',
    #     'unclear': 'unclear', 
    #     'needsguidance': 'needsguidance',
    #     'inaccurate': 'inaccurate'
    # }
    
    # feedback_type = feedback_type_mapping.get(selected_feedback, 'helpful')

    # # 构建反馈记录
    # feedback_record = {
    #     'feedback_id': f"fb-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}",
    #     'session_id': session_id,
    #     'user_query': user_query,
    #     'assistant_response': assistant_response,
    #     'selected_feedback': selected_feedback,
    #     'custom_feedback': custom_feedback,
    #     'timestamp': timestamp,
    #     'request_id': request_id
    # }

    # 记录到日志
    # logger.info(f'收到用户反馈: session_id={session_id}, selected_feedback={selected_feedback}')
    # logger.debug(f'反馈详情: {json.dumps(feedback_record, ensure_ascii=False, indent=2)}')

    # # 立即处理反馈并更新prompt
    # try:
    #     # 处理反馈 - 所有类型都会立即生效
    #     optimization_result = NativeChator_med_audio.process_feedback(
    #         session_id=session_id,
    #         feedback_type=feedback_type,
    #         custom_feedback=custom_feedback,
    #         user_query=user_query,
    #         assistant_response=assistant_response
    #     )
        
    #     # 确定优化类型
    #     if feedback_type == 'helpful':
    #         optimization_type = 'maintained'  # 保持不变
    #         msg = '反馈已收到，当前策略保持不变'
    #     elif feedback_type in ['unclear', 'needsguidance', 'inaccurate']:
    #         optimization_type = 'standard_adjustment'
    #         msg = f'反馈已收到，已应用{feedback_type}类型的标准优化'
        
    #     # 如果有具体意见，会触发额外的动态优化
    #     if custom_feedback and len(custom_feedback) > 10:
    #         optimization_type = 'custom_optimization'
    #         msg = '反馈已收到，已根据您的具体意见进行个性化优化'
        
    #     logger.info(f"Session {session_id} 应用了优化类型: {optimization_type}")
        
    # except Exception as e:
    #     logger.error(f'处理反馈优化时出错: {str(e)}')
    #     msg = '反馈已收到，但优化过程出现问题'

    # # 保存到反馈日志文件
    # try:
    #     feedback_logger = setup_logger('FEEDBACK', log_file='logs/feedback.log')
    #     feedback_record['optimization_type'] = optimization_type
    #     feedback_logger.info(json.dumps(feedback_record, ensure_ascii=False))
    # except Exception as log_error:
    #     logger.warning(f'保存反馈')
    
    return jsonify({
        'msg': "testing",
        'code': 200,
        'optimization_triggered': "123",
        'timestamp': datetime.now().isoformat()
    }), 200


@app.route('/prompt/optimization_report', methods=['GET'])
def get_optimization_report():
    """获取prompt优化报告API"""
    try:
        session_id = request.args.get('session_id', None)
        
        # 获取优化报告
        report = NativeChator_med_audio.get_optimization_report(session_id)
        
        return jsonify({
            'code': 200,
            'msg': 'success',
            'report': report,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f'获取优化报告失败: {str(e)}')
        return jsonify({
            'code': 500,
            'msg': f'获取报告失败: {str(e)}'
        }), 500


@app.route('/prompt/reset', methods=['POST'])
def reset_session_prompt():
    """重置会话prompt为默认值"""
    try:
        request_data = request.get_json()
        session_id = request_data.get('session_id')
        
        if not session_id:
            return jsonify({
                'code': 400,
                'msg': '缺少session_id参数'
            }), 400
        
        # 重置prompt
        NativeChator_med_audio.reset_session_prompt(session_id)
        
        return jsonify({
            'code': 200,
            'msg': f'Session {session_id} prompt已重置',
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f'重置prompt失败: {str(e)}')
        return jsonify({
            'code': 500,
            'msg': f'重置失败: {str(e)}'
        }), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5019, debug=False)
    

# nohup python3 -m chat_llm.chat_api_wrap &
