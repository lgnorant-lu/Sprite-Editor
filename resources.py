"""
---------------------------------------------------------------
File name:                  resources.py
Author:                     Ignorant-lu
Date created:               2025/04/15
Description:                PyQt资源文件的占位符模块，防止资源未生成时报错。
----------------------------------------------------------------

Changed history:            
                            2025/04/15: 初始创建;
----
"""

# 空的资源模块
# 此模块是资源文件的占位符
# 如需使用实际图标，请运行build_resources.py脚本

# 初始化一个空字典，避免在没有实际图标时出错
from PyQt6.QtCore import QFile
from PyQt6.QtGui import QIcon

def get_icon(name):
    """获取图标，如果不存在则返回空图标

    Args:
        name (str): 图标名称
    Returns:
        QIcon: 返回QIcon对象，若无资源则为空图标
    """
    return QIcon()

# 注册资源初始化函数，避免在没有资源时出错
def qInitResources():
    """资源初始化函数（占位）"""
    pass

def qCleanupResources():
    """资源清理函数（占位）"""
    pass 