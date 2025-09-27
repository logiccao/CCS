# -*- coding: utf-8 -*-
"""
文件名: naive_chat.py
创建时间: 2025/09/26
作者: logiccao
增强版：支持基于用户反馈的动态prompt优化 - 标准化版本
"""
import re
import json
import requests
from datetime import datetime
from collections import defaultdict
from chat_llm.chat_config import CFG as CHAT_CONF
from openai import OpenAI
from chat_llm.logger import setup_logger
logger = setup_logger('CCS', log_file='logs/CCS.log')
from chat_llm.config import api_key


# 基础prompt模板
BASE_PROMPT = """# 角色定义
你是一个医疗健康咨询助手，可以用医生的口吻简明扼要地回答用户的医疗健康问题

# 交互规范
- **语气风格**：专业医生形象，保持温暖亲切，使用拟人化口气和通俗易读的语言，减少医学术语使用。
- **回答格式**：
  • 结论优先：对于有确定结论的问题，先给出核心结论，再进行简单解释
  • 解释说明：简明扼要，避免长篇大论。
- **回答要求**
  • 是否型问题和选择型问题：回答尽量简洁；
  • 开放性问题：回答尽量简洁

# 以下为回答样例
## 样例一
用户：糖尿病可以吃西瓜吗
回答：如果血糖控制稳定的话，可以少量吃西瓜，但是要控制摄入量并监测血糖变化。如果血糖控制不好的话或者吃得多的话，有可能导致血糖迅速升高。建议每次吃西瓜不要超过100克，吃完后2小时监测一下血糖，如血糖明显升高，就尽量不要吃。

## 样例二
用户：睡觉不好可以吃褪黑素吗？
回答：可以短期吃褪黑素，来改善一下睡眠，但是尽量不要长期依赖它。褪黑素对调节睡眠节律有一定帮助，尤其适用于时差调整或短期失眠。但长期使用可能抑制自身分泌，并可能引起头晕、头痛等副作用。"""

# 标准化调整指令
ADJUSTMENT_TEMPLATES = {
    'unclear': """
# 额外清晰度增强指令
1. 使用更具体的量化表达（如"100克"代替"少量"）
2. 复杂解释采用分步说明："第一步...第二步..."
3. 关键信息重复强调，确保用户不会遗漏重点
4. 避免使用专业术语，必要时用生活化比喻解释""",
    
    'needsguidance': """
# 额外详细指导指令
1. 在基础回答后追加实用建议章节
2. 提供具体的行动步骤："明天您可以尝试..."
3. 增加简单的自我监测方法指导
4. 适当扩展解释深度，但保持核心简洁
5. 字数限制放宽：+50字用于详细指导""",
    
    'inaccurate': """
# 额外准确性保障指令
1. 采用更保守的表述："通常建议..."而非"一定可以..."
2. 自动追加免责声明："个体差异较大，建议咨询医师确认"
3. 复杂情况主动建议线下就医
4. 增加权威依据提示："根据一般医疗原则..."
5. 强调建议的局限性，避免绝对化判断"""
}


def cut_messages(messages, last_n_round = 5):
    """输入的messages，第一个为user, 最后一个也是user，保存最后一个user前面 last_n_round对话信息"""
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
    增强功能：基于用户反馈动态优化prompt - 标准化版本
    """
    def __init__(self, name = '', use_model = None, logger = None) -> None:
        self.logger = setup_logger('AURACALL', log_file='logs/chat_api.log')
        self.CHAT_CONF = CHAT_CONF
        
        # 使用标准化的基础prompt
        self.base_prompt = BASE_PROMPT
        self.system_prompt = BASE_PROMPT
        
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
        
        # 反馈和prompt管理
        self.session_prompts = {}  # session_id -> customized_prompt
        self.session_adjustments = {}  # session_id -> set of active adjustments
        self.prompt_optimization_history = []  # 保存prompt优化历史
        self.feedback_history = defaultdict(list)  # session_id -> list of feedbacks

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

    def get_session_prompt(self, session_id):
        """获取会话特定的prompt，如果没有则返回基础prompt"""
        return self.session_prompts.get(session_id, self.base_prompt)

    def chat_with_query(self, session_id, query, knowledge):
        user_assistant_history = self.get_history(session_id = session_id)
        user_assistant_history = user_assistant_history + [{'role' : 'user', 'content' : query}]
        ## 更新
        if session_id in self.conversations:
            self.conversations[session_id]['user_assistant_history'] = user_assistant_history
        else:
            self.conversations[session_id] = {'user_assistant_history' : user_assistant_history}
        
        # 使用会话特定的prompt
        resp = self.chat_with_messages(session_id, user_assistant_history=user_assistant_history, query=query, knowledge=knowledge)
        return resp 

    def retrieve_knowledge(self, query, base_url="http://101.201.212.43:8090", api_key=api_key):
        url = f"{base_url}/v1/knowledge/retrieve"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "query": query
        }
        
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()


    def store_to_history(self, session_id, full_text):
        """将最后得到的全部长度结果，存储到user_assistant_history"""
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

    def chat_with_messages(self, session_id, user_assistant_history : list, query:str, knowledge: None):
        """主函数，输入用户和助手历史对话，输出response"""
        if len(user_assistant_history) > CHAT_CONF.CONVERSATION_LAST_N_ROUND * 2:
            self.logger.info('当前对话历史过长，开始截短')
            user_assistant_history = cut_messages(user_assistant_history)
        
        if knowledge:
            knowledge = self.retrieve_knowledge(query)

        # 使用会话特定的prompt
        current_prompt = self.get_session_prompt(session_id) + f"\n检索知识：{knowledge}"
        system_message = {
            'role' : 'system',
            'content' : current_prompt
        }
        current_messages = [system_message] + user_assistant_history
        self.logger.info(f'{session_id} 当前输入大模型的用户多轮对话如下：\n {user_assistant_history}')
        self.logger.info(f'{session_id} 当前输入大模型的系统指令如下：\n {current_prompt}')

        if self.current_model == 'local':
            model = self.use_model or 'deepseek-r1' 
        elif self.current_model == 'large':
            model = self.use_model or 'deepseek-r1'
        self.logger.info(f'self.current_model : {model}')
        
        try:
            response = self.current_client.chat.completions.create(
                    model = model,
                    messages = current_messages,
                    stream=True  # 启用流式传输
                )
            self.error_counts[self.current_model] = 0
            return response 
        except Exception as e:
            self.logger.error(f'{str(e)}')
            self.handle_response_error()

    def process_feedback(self, session_id, feedback_type, custom_feedback=None, 
                        user_query=None, assistant_response=None):
        """处理用户反馈并立即更新prompt
        
        Args:
            session_id: 会话ID
            feedback_type: 反馈类型 (helpful/unclear/needsguidance/inaccurate)
            custom_feedback: 用户的具体意见
            user_query: 用户的问题
            assistant_response: 助手的回答
        """
        # 记录反馈
        feedback_record = {
            'type': feedback_type,
            'custom': custom_feedback,
            'user_query': user_query,
            'assistant_response': assistant_response,
            'timestamp': datetime.now().isoformat()
        }
        
        self.feedback_history[session_id].append(feedback_record)
        self.logger.info(f"Session {session_id} 收到反馈: {feedback_type}")
        if custom_feedback:
            self.logger.info(f"具体意见: {custom_feedback}")
        
        # 立即更新prompt
        if feedback_type == 'helpful':
            # helpful时保持不变
            self.logger.info(f"Session {session_id} 收到正面反馈，保持当前策略")
            return feedback_record
        
        # 处理标准化调整
        if feedback_type in ['unclear', 'needsguidance', 'inaccurate']:
            self.apply_standard_adjustment(session_id, feedback_type)
        
        # 如果有具体意见，立即调用大模型优化
        if custom_feedback and len(custom_feedback) > 10:
            self.logger.info(f"Session {session_id} 有具体意见，触发动态优化")
            self.optimize_prompt_with_custom_feedback(session_id, custom_feedback)
        
        return feedback_record

    def apply_standard_adjustment(self, session_id, feedback_type):
        """应用标准化的prompt调整
        
        Args:
            session_id: 会话ID
            feedback_type: 反馈类型
        """
        # 初始化会话调整集合
        if session_id not in self.session_adjustments:
            self.session_adjustments[session_id] = set()
        
        # 添加新的调整类型
        self.session_adjustments[session_id].add(feedback_type)
        
        # 重建prompt
        updated_prompt = self.base_prompt
        
        # 按顺序添加所有激活的调整
        for adjustment_type in ['unclear', 'needsguidance', 'inaccurate']:
            if adjustment_type in self.session_adjustments[session_id]:
                updated_prompt += "\n" + ADJUSTMENT_TEMPLATES[adjustment_type]
        
        # 保存更新后的prompt
        self.session_prompts[session_id] = updated_prompt
        
        # 记录优化历史
        self.prompt_optimization_history.append({
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'adjustment_type': feedback_type,
            'active_adjustments': list(self.session_adjustments[session_id]),
            'method': 'standard_adjustment'
        })
        
        self.logger.info(f"Session {session_id} 应用标准化调整: {feedback_type}")
        self.logger.debug(f"当前激活的调整: {self.session_adjustments[session_id]}")

    def optimize_prompt_with_custom_feedback(self, session_id, custom_feedback):
        """基于用户具体意见优化prompt
        
        Args:
            session_id: 会话ID
            custom_feedback: 用户的具体反馈
        """
        try:
            # 获取当前prompt
            current_prompt = self.get_session_prompt(session_id)
            
            # 构建优化请求
            optimization_prompt = f"""你是一个专业的prompt工程师。请基于用户的具体反馈优化以下医疗咨询助手的system prompt。

当前system prompt:
```
{current_prompt}
```

用户具体反馈：
{custom_feedback}

优化要求：
1. 保持原有的基础结构和格式
2. 根据用户反馈针对性地调整相关部分
3. 保持医疗专业性和温暖亲切的语气
4. 确保回答仍然简洁（是否型50字内，开放型80字内）
5. 如果用户希望更详细，可适当放宽字数限制但仍需简洁

请直接返回优化后的完整system prompt，不要包含任何解释。"""

            # 调用LLM进行优化
            response = self.current_client.chat.completions.create(
                model=self.use_model or 'deepseek-r1',
                messages=[
                    {'role': 'user', 'content': optimization_prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            optimized_prompt = response.choices[0].message.content.strip()
            
            # 验证优化结果
            if self._validate_optimized_prompt(optimized_prompt):
                # 保存优化后的prompt
                self.session_prompts[session_id] = optimized_prompt
                
                # 记录优化历史
                self.prompt_optimization_history.append({
                    'session_id': session_id,
                    'timestamp': datetime.now().isoformat(),
                    'custom_feedback': custom_feedback,
                    'method': 'llm_optimization',
                    'success': True
                })
                
                self.logger.info(f"Session {session_id} 基于具体意见的prompt优化成功")
            else:
                self.logger.warning(f"Session {session_id} 优化后的prompt验证失败，保持原prompt")
                
        except Exception as e:
            self.logger.error(f"优化prompt时出错: {str(e)}")

    def _validate_optimized_prompt(self, prompt):
        """验证优化后的prompt是否合理
        
        Args:
            prompt: 优化后的prompt
            
        Returns:
            bool: 是否通过验证
        """
        # 基本长度检查
        if len(prompt) < 100 or len(prompt) > 5000:
            return False
        
        # 确保包含关键元素
        required_keywords = ['角色定义', '交互规范', '医疗']
        for keyword in required_keywords:
            if keyword not in prompt:
                return False
        
        return True

    def get_optimization_report(self, session_id=None):
        """获取prompt优化报告
        
        Args:
            session_id: 可选，指定会话ID获取特定会话的优化历史
        
        Returns:
            优化报告字典
        """
        if session_id:
            # 返回特定会话的优化信息
            return {
                'session_id': session_id,
                'current_prompt': self.get_session_prompt(session_id),
                'active_adjustments': list(self.session_adjustments.get(session_id, [])),
                'feedback_history': self.feedback_history.get(session_id, []),
                'optimization_history': [
                    h for h in self.prompt_optimization_history 
                    if h['session_id'] == session_id
                ]
            }
        else:
            # 返回整体优化统计
            return {
                'total_optimizations': len(self.prompt_optimization_history),
                'sessions_with_custom_prompt': len(self.session_prompts),
                'recent_optimizations': self.prompt_optimization_history[-10:],
                'all_sessions': list(self.session_prompts.keys())
            }

    def reset_session_prompt(self, session_id):
        """重置会话prompt为默认值
        
        Args:
            session_id: 会话ID
        """
        if session_id in self.session_prompts:
            del self.session_prompts[session_id]
        
        if session_id in self.session_adjustments:
            del self.session_adjustments[session_id]
            
        if session_id in self.feedback_history:
            del self.feedback_history[session_id]
            
        self.logger.info(f"Session {session_id} prompt已重置为默认值")

    def get_prompt_diff(self, session_id):
        """获取会话prompt与基础prompt的差异
        
        Args:
            session_id: 会话ID
            
        Returns:
            差异说明
        """
        if session_id not in self.session_prompts:
            return "使用基础prompt，无自定义调整"
        
        adjustments = self.session_adjustments.get(session_id, set())
        custom_optimized = any(
            h.get('method') == 'llm_optimization' and h['session_id'] == session_id 
            for h in self.prompt_optimization_history
        )
        
        diff_info = {
            'has_adjustments': len(adjustments) > 0,
            'active_adjustments': list(adjustments),
            'has_custom_optimization': custom_optimized,
            'current_prompt_preview': self.session_prompts[session_id][:500] + "..."
        }
        
        return diff_info