"""
---------------------------------------------------------------
File name:                  presets.py
Author:                     Ignorant-lu
Date created:               2025/04/15
Description:                管理参数预设的类
----------------------------------------------------------------

Changed history:            
                            2025/04/15: 从sprite_mask_editor.py拆分为独立模块;
----
"""
import os
import json
import logging

class PresetManager:
    """管理参数预设的类。

    负责加载、保存、删除和列出存储在用户目录下的JSON格式预设文件。
    """
    def __init__(self, app_name):
        """初始化PresetManager。

        Args:
            app_name (str): 应用程序名称，用于确定存储预设的目录。
        """
        self.app_name = app_name
        # 在用户主目录下创建或查找应用专属的预设存储目录 (e.g., ~/.spritemaskeditor)
        self.presets_dir = os.path.join(os.path.expanduser("~"), f".{app_name.lower()}")
        os.makedirs(self.presets_dir, exist_ok=True) # 确保目录存在

    def get_presets_list(self):
        """获取所有已保存预设的名称列表。

        扫描预设目录下的所有.json文件。

        Returns:
            list[str]: 预设名称列表 (不含.json后缀)。
        """
        result = []
        try:
            for file in os.listdir(self.presets_dir):
                if file.endswith(".json"):
                    result.append(file[:-5]) # 去掉.json后缀
        except Exception as e:
            logging.error(f"获取预设列表出错: {e}")
        return sorted(result) # 返回排序后的列表

    def save_preset(self, name, params):
        """将参数保存为指定名称的预设文件。

        Args:
            name (str): 预设名称。
            params (dict): 要保存的参数字典。

        Returns:
            bool: 如果保存成功则返回True，否则返回False。
        """
        # 对预设名称进行基本清理，防止路径问题
        safe_name = "".join(c for c in name if c.isalnum() or c in ('_', '-')).rstrip()
        if not safe_name:
             logging.error(f"无效的预设名称 '{name}'")
             return False
             
        try:
            file_path = os.path.join(self.presets_dir, f"{safe_name}.json")
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(params, f, ensure_ascii=False, indent=2) # 保存为格式化的JSON
            return True
        except Exception as e:
            logging.exception(f"保存预设 '{safe_name}' 出错: {e}")
            return False

    def load_preset(self, name):
        """加载指定名称的预设文件。

        Args:
            name (str): 预设名称。

        Returns:
            dict | None: 加载的参数字典，如果文件不存在或加载失败则返回None。
        """
        try:
            file_path = os.path.join(self.presets_dir, f"{name}.json")
            if not os.path.exists(file_path):
                logging.warning(f"预设文件不存在: {file_path}")
                return None
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logging.exception(f"加载预设 '{name}' 出错: {e}")
            return None

    def delete_preset(self, name):
        """删除指定名称的预设文件。

        Args:
            name (str): 预设名称。

        Returns:
            bool: 如果删除成功或文件不存在则返回True，否则返回False。
        """
        file_path = os.path.join(self.presets_dir, f"{name}.json")
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
            return True # 文件不存在也视为成功删除
        except Exception as e:
            logging.error(f"删除预设 '{name}' 出错: {e}")
            return False
