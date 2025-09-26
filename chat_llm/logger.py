# -*- coding: utf-8 -*-
"""
文件名: logger.py
创建时间: 2025/07/16
作者: logiccao
"""
import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler


def setup_logger(name=None, log_level=logging.INFO, log_file=None):
    """
    创建和配置logger
    
    Args:
        name: logger名称, 默认为调用模块名
        log_level: 日志级别
        log_file: 日志文件路径, 如果为None则使用默认路径
    """
    # 使用调用模块名作为logger名称
    if name is None:
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'unknown')
    
    logger = logging.getLogger(name)
    
    # 避免重复添加handler
    if logger.handlers:
        return logger
        
    logger.setLevel(log_level)
    
    # 创建logs目录
    logs_dir = 'logs'
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    # 默认日志文件路径
    if log_file is None:
        log_file = os.path.join(logs_dir, f'{name.split(".")[-1]}.log')
    
    # 创建轮转文件处理器 (最大10MB，保留5个文件)
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(log_level)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    # 创建格式器
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    file_handler.setFormatter(detailed_formatter)
    console_handler.setFormatter(simple_formatter)
    
    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


if __name__ == "__main__":
    # 创建主logger
    pass
    # main_logger = setup_logger('BQServer', log_file='logs/BQServer.log')

