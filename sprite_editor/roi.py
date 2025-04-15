"""
---------------------------------------------------------------
File name:                  roi.py
Author:                     Ignorant-lu
Date created:               2025/04/15
Description:                定义表示提取帧的ROI（感兴趣区域）的类
----------------------------------------------------------------

Changed history:            
                            2025/04/15: 从sprite_mask_editor.py拆分为独立模块;
----
"""
import numpy as np

class FrameROI:
    """表示单个提取的角色帧（Region of Interest）。

    包含图像数据、蒙版、位置、大小、面积、索引、标签、备注以及蒙版编辑历史等信息。

    Attributes:
        img (np.ndarray): 帧图像数据 (RGBA格式)。
        mask (np.ndarray): 帧的原始或编辑后的蒙版数据 (灰度图)。
        x (int): 帧在原始图像中的左上角x坐标。
        y (int): 帧在原始图像中的左上角y坐标。
        w (int): 帧的宽度。
        h (int): 帧的高度。
        area (float): 帧的轮廓面积。
        idx (int): 帧的唯一索引号。
        aspect_ratio (float): 帧的长宽比。
        selected (bool): 帧是否被选中。
        tag (str): 用户定义的标签。
        note (str): 用户定义的备注。
        mask_edit_history (list): 蒙版编辑历史记录。
        mask_edit_idx (int): 当前蒙版历史记录的索引。
    """
    def __init__(self, img, mask, x, y, w, h, area, idx, tag="", note=""):
        """初始化FrameROI对象。

        Args:
            img (np.ndarray): 帧图像数据。
            mask (np.ndarray): 帧的蒙版数据。
            x (int): x坐标。
            y (int): y坐标。
            w (int): 宽度。
            h (int): 高度。
            area (float): 面积。
            idx (int): 索引号。
            tag (str, optional): 标签. Defaults to "".
            note (str, optional): 备注. Defaults to "".
        """
        self.img = img
        self.mask = mask
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.area = area
        self.idx = idx
        self.aspect_ratio = w / h if h > 0 else 0
        self.selected = False
        self.tag = tag
        self.note = note
        self.mask_edit_history = [mask.copy()]
        self.mask_edit_idx = 0
        
    def add_mask_to_history(self, mask):
        """添加新的mask到历史记录"""
        if self.mask_edit_idx < len(self.mask_edit_history) - 1:
            self.mask_edit_history = self.mask_edit_history[:self.mask_edit_idx + 1]
        self.mask_edit_history.append(mask.copy())
        self.mask_edit_idx = len(self.mask_edit_history) - 1
        
    def get_current_mask(self):
        """获取当前历史记录中的mask"""
        return self.mask_edit_history[self.mask_edit_idx].copy()
        
    def undo_mask(self):
        """撤销mask修改"""
        if self.mask_edit_idx > 0:
            self.mask_edit_idx -= 1
            self.mask = self.mask_edit_history[self.mask_edit_idx].copy()
            return True
        return False
        
    def redo_mask(self):
        """重做mask修改"""
        if self.mask_edit_idx < len(self.mask_edit_history) - 1:
            self.mask_edit_idx += 1
            self.mask = self.mask_edit_history[self.mask_edit_idx].copy()
            return True
        return False
        
    def reset_mask(self):
        """重置为原始mask"""
        if self.mask_edit_history:
            self.mask_edit_idx = 0
            self.mask = self.mask_edit_history[0].copy()
            return True
        return False
