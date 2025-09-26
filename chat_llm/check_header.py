
from flask import request, jsonify, g
import uuid
from chat_llm.chat_config import CFG
import uuid
users = {
    "user1": CFG.api_authorization_token
}


from flask import request, abort
def check_header(app):
    @app.before_request
    def _check_header():
        if request.method == 'OPTIONS':
            return 
        token = request.headers.get("Authorization")
        if token is None:
            abort(401)
        if token not in users.values():
            abort(403)
    

from flask import request, abort
def check_header(app):
    @app.before_request
    def _check_header():
        # print(request)
        if request.method == 'OPTIONS':
            return 
        token = request.headers.get(CFG.api_authorization_head_key)
        if token is None:
            abort(401)
        if token not in users.values():
            abort(403)
        
        request_id = request.headers.get('request_id')
        if request_id is None:
        # 如果请求头中没有request_id，生成一个新的
            request_id = str(uuid.uuid4())
        # 将request_id存储在g对象中
        g.request_id = request_id
        # _check_header()