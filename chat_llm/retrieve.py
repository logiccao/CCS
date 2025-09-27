# -*- coding: utf-8 -*-
"""
文件名: knowledge_retiravl.py
创建时间: 2025/09/27 16:59:05
作者: logiccao
"""
import requests
from chat_llm.config import api_key


def retrieve_knowledge(query, base_url="http://101.201.212.43:8090", api_key=api_key):
    """
    查询知识库
    
    Args:
        query (str): 查询内容
        base_url (str): API服务器地址
        api_key (str): API密钥
        
    Returns:
        dict: API响应结果
    """
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


# 使用示例
if __name__ == "__main__":
    result = retrieve_knowledge("新生儿科在哪里")
    
    print(f"查询成功: ")
    print(result)

