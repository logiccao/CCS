# -*- coding: utf-8 -*-
"""
文件名: app_ccs.py
创建时间: 2025/09/28
作者: logiccao
"""
from fastapi import FastAPI, Request, Response, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, FileResponse
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
import os
import time
import json
import secrets
from typing import Annotated
import uvicorn
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，生产环境应指定具体域名
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头
)

security = HTTPBasic()

# 配置
UPLOAD_FOLDER = 'uploads'
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
ALLOWED_EXTENSIONS = {'pcm'}
PASSWORD = "369.logic"  # 生产环境应从安全来源获取

# 配置模板
templates = Jinja2Templates(directory="templates")

# 挂载静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")

# 添加 HTTPS 重定向中间件
app.add_middleware(HTTPSRedirectMiddleware)

# 会话状态管理（使用内存中的简单实现）
sessions = {}

def get_session(request: Request):
    """获取或创建会话"""
    session_id = request.cookies.get("session_id")
    if session_id not in sessions:
        session_id = secrets.token_urlsafe(24)
        sessions[session_id] = {"authenticated": False}
    return session_id, sessions[session_id]

def is_authenticated(request: Request):
    """检查用户是否已认证"""
    _, session_data = get_session(request)
    return session_data.get("authenticated", False)

# 修复方案1：抛出异常来中断请求
async def login_required(request: Request):
    """认证保护依赖项 - 抛出异常版本"""
    if not is_authenticated(request):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="需要登录",
            headers={"Location": "/login"}
        )
    return None

# 修复方案2：使用自定义异常处理器（推荐）
class AuthenticationRequired(Exception):
    pass

@app.exception_handler(AuthenticationRequired)
async def auth_exception_handler(request: Request, exc: AuthenticationRequired):
    return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)

async def login_required_v2(request: Request):
    """认证保护依赖项 - 自定义异常版本（推荐使用这个）"""
    if not is_authenticated(request):
        raise AuthenticationRequired()
    return None

def allowed_file(filename: str):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# 登录页面
@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    if is_authenticated(request):
        return RedirectResponse(url="/ccs")
    return templates.TemplateResponse("login.html", {"request": request, "error": None})

@app.post("/login")
async def login(request: Request, password: str = Form(...)):
    if password == PASSWORD:
        session_id, session_data = get_session(request)
        session_data["authenticated"] = True
        response = RedirectResponse(url="/ccs", status_code=status.HTTP_303_SEE_OTHER)
        response.set_cookie(
            key="session_id",
            value=session_id,
            httponly=True,
            secure=True,
            samesite="lax"
        )
        return response
    
    return templates.TemplateResponse(
        "login.html", 
        {"request": request, "error": "密码错误，请重试"},
        status_code=status.HTTP_401_UNAUTHORIZED
    )

# 登出
@app.get("/logout")
async def logout(request: Request):
    session_id = request.cookies.get("session_id")
    if session_id in sessions:
        sessions.pop(session_id)
    response = RedirectResponse(url="/login")
    response.delete_cookie("session_id")
    return response

# 首页
@app.get("/")
async def home():
    return RedirectResponse(url="/login")

# 语音病历 - 使用修复后的认证保护
@app.get("/ccs")
async def serve_index(request: Request, _: None = Depends(login_required_v2)):
    return FileResponse("templates/index.html")


# 住院病历 - 使用修复后的认证保护
@app.get("/inpatientrecord")
async def serve_index_inpatient(request: Request, _: None = Depends(login_required_v2)):
    return FileResponse("templates/inpatientRecord.html")

# 音频上传 - 也需要认证保护
@app.post("/htt/audioupload")
async def handle_audio(
    request: Request,
    audio: UploadFile = File(...),
    sample_rate: int = Form(16000),
    bit_depth: int = Form(16),
    channels: int = Form(1),
    _: None = Depends(login_required_v2)  # 添加认证保护
):
    start_time = time.time()
    
    # 验证文件
    if not allowed_file(audio.filename):
        raise HTTPException(400, detail="只允许上传 .pcm 文件")
    
    # 安全文件名处理
    original_filename = audio.filename.replace("\\", "/").split("/")[-1]
    base_name = os.path.splitext(original_filename)[0]
    pcm_path = os.path.join(UPLOAD_FOLDER, original_filename)
    
    try:
        # 保存文件
        with open(pcm_path, "wb") as buffer:
            content = await audio.read()
            buffer.write(content)
        
        # 获取文件信息
        file_size = os.path.getsize(pcm_path)
        processing_time = time.time() - start_time
        
        return JSONResponse({
            "status": "success",
            "original_file": original_filename,
            "processing_time_sec": round(processing_time, 2)
        })
    
    except Exception as e:
        raise HTTPException(500, detail=str(e))

# 创建登录模板
def create_login_template():
    template_dir = Path("templates")
    template_dir.mkdir(exist_ok=True)
    
    login_html = """
    <!DOCTYPE html>
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
    </html>
    """
    
    with open(template_dir / "login.html", "w", encoding="utf-8") as f:
        f.write(login_html)

# 创建模板文件（如果不存在）
if not Path("templates/login.html").exists():
    create_login_template()

if __name__ == "__main__":
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=51218,
        ssl_keyfile="zhengshu/sophonine.com.key",
        ssl_certfile="zhengshu/sophonine.com_bundle.pem",
        reload=False
    )
    server = uvicorn.Server(config)
    server.run()
