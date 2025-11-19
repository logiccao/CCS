# -*- coding: utf-8 -*-
"""
文件名: app_ccs.py
创建时间: 2025/09/28
作者: logiccao
"""
import re
import time
import uuid
import json
import secrets
import uvicorn
from pathlib import Path
from typing import Union, Optional
from datetime import datetime
from fastapi import FastAPI, Request, Depends, HTTPException, status, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# 导入业务模块
from chat_llm.logger import setup_logger
from chat_llm.naive_chat import NativeChat

# 初始化
logger = setup_logger('CCS', log_file='logs/CCS.log')
NativeChator_med_audio = NativeChat(name='naive_med_audio', logger=logger, use_model='deepseek-v3')
app = FastAPI()

# 配置中间件
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# 配置常量
PASSWORD = "369.logic"

# 配置模板
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# 业务数据存储
session_cache = {}
auth_sessions = {}

# 工具函数
def generate_session_id():
    return f"sid-{datetime.now().strftime('%Y%m%d%H%M')}-{uuid.uuid4().hex[:8]}"

end_pattern = re.compile(r'(?:^|[\s,，.。;；])(?:(?:好的|行|明白了|知道了|没问题|ok)\W*(?:谢谢|感谢|thx|3q)\W*(?:您|你)?\W*(?:再见|拜拜|结束)?[!！。.？?]*$|(?:那就这样|那就到这里|没有(?:其他)?问题了?|我(?:的)?问题(?:解决)?了|不需要了?|可以了?|没事了?|就这样吧)[!！。.？?]*$|(?:谢谢|感谢|多谢|thx|3q)\W*(?:您|你)?\W*(?:啊|啦|呢)?[!！。.？?]*$|(?:再见|拜拜|结束|挂了吧?|停吧?|bye)[\s\W]*$|(?:thank\s*you|thanks|bye|byebye)[\s\W]*$)', re.IGNORECASE)

def should_end_call(user_input):
    return bool(end_pattern.search(user_input))

# 请求模型
class ChatRequest(BaseModel):
    age: int = 0
    sex: str
    query: str
    session_id: str = ""
    dialog_type: str = ""
    dialog_mode: str = ""

class FeedbackRequest(BaseModel):
    sessionId: Optional[str] = None
    userQuery: str
    assistantResponse: str
    customFeedback: str
    dialogMode: str
    dialogType: str
    feedbackType: str
    problemSolved: Union[str, bool]  # 允许字符串或布尔值
    rating: Union[str, int]          # 允许字符串或整数
    timestamp: str = ""

class SessionRequest(BaseModel):
    session_id: str

# 认证相关
def get_auth_session(request: Request):
    session_id = request.cookies.get("session_id")
    if session_id not in auth_sessions:
        session_id = secrets.token_urlsafe(24)
        auth_sessions[session_id] = {"authenticated": False}
    return session_id, auth_sessions[session_id]

def is_authenticated(request: Request):
    _, session_data = get_auth_session(request)
    return session_data.get("authenticated", False)

class AuthenticationRequired(Exception):
    pass

@app.exception_handler(AuthenticationRequired)
async def auth_exception_handler(request: Request, exc: AuthenticationRequired):
    return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)

async def require_auth(request: Request):
    if not is_authenticated(request):
        raise AuthenticationRequired()
    return None

# 聊天API
@app.post('/chat_audio/naive_med')
async def naive_med_chat_api(chat_request: ChatRequest):
    request_id = str(uuid.uuid4().hex[:8])
    logger.debug(f'request_id: {request_id}')
    
    query = chat_request.query
    session_id = chat_request.session_id
    
    if 'multi' == chat_request.dialog_mode:
        logger.info("多轮对话模式")
        if not session_id:
            session_id = generate_session_id()
            session_cache[session_id] = ''
            logger.info('新会话')
        else:
            if session_id not in session_cache:
                raise HTTPException(status_code=400, detail='请求错误：session_id')
            elif session_cache.get(session_id) == 'done':
                raise HTTPException(status_code=400, detail='请求错误：会话已结束')

        logger.info(f'request data: {chat_request.model_dump()}')
        knowledge = chat_request.dialog_type == "knowledge"
        stream_resp = NativeChator_med_audio.chat_with_query(session_id=session_id, query=query, knowledge=knowledge)
    else:
        knowledge = chat_request.dialog_type == "knowledge"
        stream_resp = NativeChator_med_audio.chat_with_query_single(query=query, knowledge=knowledge)
    
    full_text = ''
    session_finish = 'false'
    
    def generate_stream():
        nonlocal full_text, session_finish
        for chunk in stream_resp:  # 同步遍历流式响应
            if chunk:
                for choice in chunk.choices:
                    content = choice.delta.content
                    if content:
                        full_text += content
                        response_data = {
                            "text_chunk": content,
                            "session_finish": session_finish,
                            "session_id": session_id
                        }
                        yield f'id: {request_id}\nevent: message\ndata: {json.dumps(response_data)}\n\n'
        
        # 处理流式结束
        if full_text:
            end_call = should_end_call(query)
            session_finish = 'true' if end_call else 'false'
            if session_finish == 'true':
                session_cache[session_id] = 'done'
            response_data = {
                "text_chunk": '',
                "session_finish": session_finish,
                "session_id": session_id
            }
            yield f'id: {request_id}\nevent: done\ndata: {json.dumps(response_data)}\n\n'
        
        # 在生成器结束后记录
        logger.debug(f'request_id: {request_id}')
        if 'multi' == chat_request.dialog_mode:
            NativeChator_med_audio.store_to_history(session_id, full_text)
        logger.debug(f'模型全部回答：{full_text}')

    return StreamingResponse(generate_stream(), media_type="text/event-stream")

@app.post('/session/history')
async def session_history(history_request: SessionRequest):
    try:
        session_id = history_request.session_id
        if hasattr(NativeChator_med_audio, 'conversations'):
            conversations = NativeChator_med_audio.conversations
        else:
            return {'msg': 'success', 'history': []}
            
        if session_id not in conversations:
            msg = f'{session_id} not in conversations'
        elif 'user_assistant_history' not in conversations[session_id]:
            msg = f'{session_id} has no user_assistant_history'
        else:
            return {'msg': 'success', 'history': conversations[session_id]['user_assistant_history']}
        
        return {'msg': msg, 'history': []}
    except Exception as e:
        logger.error(f'api internal error: {str(e)}')
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/feedback')
async def feedback_api(feedback_request: FeedbackRequest):
    request_id = str(uuid.uuid4().hex[:8])
    logger.debug(f'feedback request_id: {request_id}')
    
    if not feedback_request.timestamp:
        feedback_request.timestamp = datetime.now().isoformat()
    
    """
    sessionId: str
    userQuery: str
    assistantResponse: str
    customFeedback: str
    dialogMode: str
    dialogType: str
    feedbackType: str
    problemSolved: Union[str, bool]  # 允许字符串或布尔值
    rating: Union[str, int]          # 允许字符串或整数
    timestamp: str = ""
    """
    # 记录反馈信息
    logger.info(f"sessionId: {feedback_request.sessionId}")
    logger.info(f"userQuery: '[{feedback_request.userQuery}]'")
    logger.info(f"assistantResponse: '[{feedback_request.assistantResponse}]'")
    logger.info(f"dialogMode: {feedback_request.dialogMode}")
    logger.info(f"dialogType: {feedback_request.dialogType}")
    logger.info(f"feedbackType: {feedback_request.feedbackType}")
    logger.info(f"problemSolved: {feedback_request.problemSolved}")
    logger.info(f"rating: {feedback_request.rating}")
    logger.info(f"customFeedback: {feedback_request.customFeedback}")
    
    if 'correction' == feedback_request.feedbackType:
        logger.info("信息纠正, 动态更新")
        if feedback_request.customFeedback:
            knowledge = f"用户提问: {feedback_request.userQuery}, 当前系统回答: '[{feedback_request.assistantResponse}]', 用户纠正知识:{feedback_request.customFeedback}, 纠正时间:{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}"
            NativeChator_med_audio.dynamic_knowledge.append(knowledge)
        else:
            logger.info("其他的情况")
    elif 'general' == feedback_request.feedbackType:
        logger.info("一般反馈,暂不处理")
        logger.info(feedback_request.customFeedback)
    else:
        logger.info("其他反馈情况")

    return {
        'msg': "success",
        'code': 200,
        'optimization_triggered': "ok",
        'timestamp': datetime.now().isoformat()
    }

# 认证路由
@app.get("/login")
async def login_page(request: Request):
    if is_authenticated(request):
        return RedirectResponse(url="/ccs")
    return templates.TemplateResponse("login.html", {"request": request, "error": None})

@app.post("/login")
async def login(request: Request, password: str = Form(...)):
    if password == PASSWORD:
        session_id, session_data = get_auth_session(request)
        session_data["authenticated"] = True
        response = RedirectResponse(url="/ccs", status_code=status.HTTP_303_SEE_OTHER)
        response.set_cookie(key="session_id", value=session_id, httponly=True, secure=True, samesite="lax")
        return response
    
    return templates.TemplateResponse("login.html", {"request": request, "error": "密码错误，请重试"}, status_code=status.HTTP_401_UNAUTHORIZED)

@app.get("/logout")
async def logout(request: Request):
    session_id = request.cookies.get("session_id")
    if session_id in auth_sessions:
        auth_sessions.pop(session_id)
    response = RedirectResponse(url="/login")
    response.delete_cookie("session_id")
    return response

# 受保护的路由
@app.get("/")
async def home():
    return RedirectResponse(url="/login")

@app.get("/ccs")
async def serve_index(request: Request, _: None = Depends(require_auth)):
    return FileResponse("templates/index.html")

# 创建登录模板
def create_login_template():
    template_dir = Path("templates")
    template_dir.mkdir(exist_ok=True)
    
    login_html = """<!DOCTYPE html>
<html>
<head>
    <title>口令验证</title>
    <style>
        body { font-family: Arial, sans-serif; background-color: #f5f5f5; }
        .login-container { max-width: 400px; margin: 100px auto; padding: 20px; background: white; border-radius: 5px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        h1 { text-align: center; color: #333; }
        input[type="password"] { width: 100%; padding: 10px; margin: 10px 0; border: 1px solid #ddd; border-radius: 4px; }
        button { width: 100%; padding: 10px; background-color: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer; }
        .error { color: red; text-align: center; }
    </style>
</head>
<body>
    <div class="login-container">
        <h1>请输入口令</h1>
        {% if error %}
            <p class="error">{{ error }}</p>
        {% endif %}
        <form method="POST">
            <input type="password" name="password" placeholder="输入口令" required>
            <button type="submit">验证</button>
        </form>
    </div>
</body>
</html>"""
    
    with open(template_dir / "login.html", "w", encoding="utf-8") as f:
        f.write(login_html)

# 初始化模板
if not Path("templates/login.html").exists():
    create_login_template()

if __name__ == "__main__":
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=51218,
        ssl_keyfile="zhengshu/eh-med.com.key",
        ssl_certfile="zhengshu/eh-med.com.pem",
        reload=False
    )
    server = uvicorn.Server(config)
    server.run()

