# -*- coding: utf-8 -*-
"""
文件名: naive_chat.py
创建时间: 2025/09/26
作者: logiccao
"""
import re
from chat_llm.chat_config import CFG as CHAT_CONF
from openai import OpenAI
from .naive_med_prompt import naive_med_prompt_v1 as naive_med_prompt
from chat_llm.logger import setup_logger
logger = setup_logger('CCS', log_file='logs/CCS.log')


def cut_messages(messages, last_n_round = 5):
    """输入的messages，第一个为user, 最后一个也是user，保存最后一个user前面 last_n_round对话信息
    """
    if len(messages) > 10:
        first_msg, last_msg = messages[0], messages[-1]
        first_role, last_role = first_msg['role'], last_msg['role']
        assert first_role == 'user'
        assert last_role == 'user'
        last_index = last_n_round * 2
        last_n_messages = messages[(-1 - last_index):] # 10的话是11 保证第一个是user
        assert last_n_messages[0]['role'] == 'user'
        return last_n_messages 
    return messages


class NativeChat(object):
    """构建基于大模型封装的原生聊天模型
    先定义模型client，再定义system_prompt，接收传入的user的user_assistant_history，产生流式输出
    使用历史几轮的逻辑在这里处理
    """
    def __init__(self, name = '', use_model = None, logger = None) -> None:
        self.logger = setup_logger('AURACALL', log_file='logs/chat_api.log')
        self.system_prompt = CHAT_CONF.SYSTEM_TEMPLATE
        self.CHAT_CONF = CHAT_CONF
        self.system_prompt = naive_med_prompt
        self.current_model = CHAT_CONF.PRIOR_MODEL 
        self.use_model = use_model
        self.error_counts = {
            'local' : 0,
            'large' : 0
        }
        self.define_client(CHAT_CONF)

        if CHAT_CONF.PRIOR_MODEL == 'local':
            self.current_client = self.client_local
        elif CHAT_CONF.PRIOR_MODEL == 'large':
            self.current_client = self.client_large
        self.logger.info(f'init chain : {CHAT_CONF.PRIOR_MODEL}')
        self.conversations = {}

    def define_client(self, CHAT_CONF = None, api_source = ''):
        if CHAT_CONF is None:
            CHAT_CONF = self.CHAT_CONF
        self.client_local = OpenAI(
            api_key = CHAT_CONF.API_KEY,
            base_url = CHAT_CONF.BASE_URL)
        self.client_large = OpenAI(
            api_key = CHAT_CONF.zzz_api_key,
            base_url = CHAT_CONF.zzz_base_url)
        if api_source == 'ali':
           self.client_large = OpenAI(
            api_key = CHAT_CONF.ali_api_key,
            base_url = CHAT_CONF.ali_base_url)         

    
    def get_history(self, session_id):
        user_assistant_history = self.conversations.get(session_id, {}).get('user_assistant_history', [])
        return user_assistant_history


    def chat_with_query(self, session_id, query):
        user_assistant_history = self.get_history(session_id = session_id)
        user_assistant_history = user_assistant_history + [{'role' : 'user', 'content' : query}]
        ## 更新
        if session_id in self.conversations:
            self.conversations[session_id]['user_assistant_history'] = user_assistant_history
        else:
            self.conversations[session_id] = {'user_assistant_history' : user_assistant_history}
        resp = self.chat_with_messages(session_id, user_assistant_history=user_assistant_history)
        return resp 

    def store_to_history(self, session_id, full_text):
        """将最后得到的全部长度结果，存储到user_assistant_history
        """
        user_assistant_history = self.get_history(session_id = session_id)
        user_assistant_history = user_assistant_history +  [{'role' : 'assistant', 'content' : full_text}]
        self.conversations[session_id]['user_assistant_history'] = user_assistant_history


    def handle_response_error(self):
        self.error_counts[self.current_model] += 1
        if self.error_counts[self.current_model] >= 2:
            self.logger.warning(f'当前模型：{self.current_model}错误次数大于2，切换模型')
            self.current_model = 'local' if self.current_model == 'large' else 'large'
            self.current_client = self.client_local if self.current_model == 'local' else self.client_large
            self.error_counts[self.current_model] = 0 


    def chat_with_messages(self, session_id, user_assistant_history : list):
        """主函数，输入用户和助手历史对话，输出response
        user_assistant_history : 输入的messages列表，第一个为user, 最后一个也是user
        """

        if len(user_assistant_history) > CHAT_CONF.CONVERSATION_LAST_N_ROUND * 2:
            self.logger.info('当前对话历史过长，开始截短')
            user_assistant_history = cut_messages(user_assistant_history)

        system_message = {
            'role' : 'system',
            'content' : self.system_prompt
        }
        current_messages = [system_message] + user_assistant_history
        self.logger.info(f'{session_id} 当前输入大模型的用户多轮对话如下：\n {user_assistant_history}')
        self.logger.info(f'{session_id} 当前输入大模型的系统指令如下：\n {self.system_prompt[:200]}')

        if self.current_model == 'local':
            model = self.use_model or 'deepseek-r1' 
        elif self.current_model == 'large':
            model = self.use_model or 'deepseek-r1'
        self.logger.info(f'self.current_model : {model}')
        
        try:
            response = self.current_client.chat.completions.create(
                    # model="ollama/deepseek-r1:70b",
                    model = model,
                    messages = current_messages,
                    stream=True  # 启用流式传输
                )
            # self.logger.info(f'大模型输出响应结果: {response}')
            self.error_counts[self.current_model] = 0
            return response 
        except Exception as e:
            self.logger.error(f'{str(e)}')
            self.handle_response_error()

