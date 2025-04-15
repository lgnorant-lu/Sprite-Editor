"""
---------------------------------------------------------------
File name:                  mask_processor.py
Author:                     Ignorant-lu
Date created:               2025/04/15
Description:                蒙版生成、ROI提取和相关工具函数
----------------------------------------------------------------

Changed history:            
                            2025/04/15: 从sprite_mask_editor.py拆分为独立模块;
----
"""
import numpy as np
import cv2
from .roi import FrameROI
import re
import logging
import os

class MaskProcessor:
    """处理蒙版生成和ROI提取的核心逻辑。

    封装了蒙版处理的算法和相关参数，与UI界面解耦。
    允许通过调整参数来控制蒙版生成和ROI提取的行为。

    Attributes:
        color_thresh (int): 色差阈值，用于区分前景和背景。
        pad (int): 提取ROI时向外扩展的像素数。
        kernel_size (int): 形态学操作（闭/开运算）的核大小。
        max_extract (int): 最多提取的ROI数量。
        min_area (int): ROI的最小面积阈值。
        max_area (int): ROI的最大面积阈值。
        close_iter (int): 闭运算的迭代次数。
        open_iter (int): 开运算的迭代次数。
        out_width (int): 输出帧的画布宽度。
        out_height (int): 输出帧的画布高度。
    """
    def __init__(self, color_thresh=40, pad=4, kernel_size=5, max_extract=12, min_area=500, max_area=1_000_000, close_iter=2, open_iter=1, out_width=128, out_height=128):
        """初始化MaskProcessor。

        Args:
            color_thresh (int, optional): 色差阈值. Defaults to 40.
            pad (int, optional): 边界扩展像素数. Defaults to 4.
            kernel_size (int, optional): 形态学核大小. Defaults to 5.
            max_extract (int, optional): 最大提取帧数. Defaults to 12.
            min_area (int, optional): 最小面积阈值. Defaults to 500.
            max_area (int, optional): 最大面积阈值. Defaults to 1_000_000.
            close_iter (int, optional): 闭运算次数. Defaults to 2.
            open_iter (int, optional): 开运算次数. Defaults to 1.
            out_width (int, optional): 输出宽度. Defaults to 128.
            out_height (int, optional): 输出高度. Defaults to 128.
        """
        self.color_thresh = color_thresh
        self.pad = pad
        self.kernel_size = kernel_size
        self.max_extract = max_extract
        self.min_area = min_area
        self.max_area = max_area
        self.close_iter = close_iter
        self.open_iter = open_iter
        self.out_width = out_width
        self.out_height = out_height

    def get_bg_samples(self, img_np):
        """从图像的边缘和中心采样背景颜色。

        Args:
            img_np (np.ndarray): 输入图像 (Numpy数组, RGB格式)。

        Returns:
            np.ndarray: 背景颜色样本数组。
        """
        h, w = img_np.shape[0], img_np.shape[1]
        return np.array([
            img_np[0,0,:3], img_np[0,-1,:3], img_np[-1,0,:3], img_np[-1,-1,:3],
            img_np[h//2,0,:3], img_np[0,w//2,:3], img_np[h//2,w//2,:3]
        ])

    def gen_mask(self, img_np):
        """根据背景颜色采样生成前景蒙版。

        使用颜色距离和形态学操作（闭运算和开运算）来生成和优化蒙版。

        Args:
            img_np (np.ndarray): 输入图像 (Numpy数组, RGBA格式)。

        Returns:
            np.ndarray: 生成的前景蒙版 (灰度图, 0或255)。
        """
        h, w = img_np.shape[0], img_np.shape[1]
        bg_samples = self.get_bg_samples(img_np)
        flat_img = img_np[...,:3].reshape(-1,3) # 忽略Alpha通道进行颜色比较
        dist = np.min(np.linalg.norm(flat_img[:,None,:] - bg_samples[None,:,:], axis=2), axis=1)
        mask_fg = (dist > self.color_thresh).astype(np.uint8).reshape(h, w) * 255
        kernel = np.ones((int(self.kernel_size),int(self.kernel_size)), np.uint8)
        mask_fg = cv2.morphologyEx(mask_fg, cv2.MORPH_CLOSE, kernel, iterations=self.close_iter)
        mask_fg = cv2.morphologyEx(mask_fg, cv2.MORPH_OPEN, kernel, iterations=self.open_iter)
        return mask_fg

    def extract_rois(self, img_np, mask_fg):
        """从前景蒙版中提取感兴趣区域（ROIs）。

        查找轮廓，根据面积过滤，并根据设置的输出尺寸自动居中。

        Args:
            img_np (np.ndarray): 原始图像 (Numpy数组, RGBA格式)。
            mask_fg (np.ndarray): 前景蒙版 (灰度图)。

        Returns:
            list[FrameROI]: 提取的FrameROI对象列表。
        """
        contours, _ = cv2.findContours(mask_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        rois = []
        idx = 1
        for cnt in contours[:self.max_extract]:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area:
                continue
            x, y, w2, h2 = cv2.boundingRect(cnt)
            pad = self.pad
            x = max(0, x - pad)
            y = max(0, y - pad)
            w2 = min(img_np.shape[1] - x, w2 + 2 * pad)
            h2 = min(img_np.shape[0] - y, h2 + 2 * pad)
            
            # 提取带透明通道的ROI图像
            roi_img_with_alpha = img_np[y:y+h2, x:x+w2].copy()
            
            # 提取对应的蒙版区域
            roi_mask = mask_fg[y:y+h2, x:x+w2]
            
            # 将蒙版应用到ROI图像的Alpha通道
            roi_img_with_alpha[...,3] = roi_mask 
            
            # 自动居中到指定画布
            out_h, out_w = self.out_height, self.out_width
            canvas = np.zeros((out_h, out_w, 4), dtype=np.uint8)
            ry, rx = roi_img_with_alpha.shape[0], roi_img_with_alpha.shape[1]
            sy = max((out_h-ry)//2, 0)
            sx = max((out_w-rx)//2, 0)
            ey = min(sy+ry, out_h)
            ex = min(sx+rx, out_w)
            cy = min(ry, out_h-sy)
            cx = min(rx, out_w-sx)
            canvas[sy:sy+cy, sx:sx+cx] = roi_img_with_alpha[:cy, :cx]
            
            # 创建FrameROI对象，注意传递的是居中后的canvas和原始未缩放的roi_mask
            rois.append(FrameROI(canvas, roi_mask, x, y, w2, h2, area, idx))
            idx += 1
        return rois

    def get_params(self):
        """获取当前所有处理参数。

        Returns:
            dict: 包含所有参数的字典。
        """
        return {
            "color_thresh": self.color_thresh,
            "pad": self.pad,
            "kernel_size": self.kernel_size,
            "max_extract": self.max_extract,
            "min_area": self.min_area,
            "max_area": self.max_area,
            "close_iter": self.close_iter,
            "open_iter": self.open_iter,
            "out_width": self.out_width,
            "out_height": self.out_height
        }

    def set_params(self, params):
        """根据提供的字典设置处理参数。

        Args:
            params (dict): 包含参数键值对的字典。
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

# Placeholder mapping for render_filename
PLACEHOLDER_MAP = {
    '[索引]': 'idx',
    '[X]': 'x',
    '[Y]': 'y',
    '[宽]': 'w',
    '[高]': 'h',
    '[面积]': 'area',
    '[长宽比]': 'aspect_ratio',
    '[标签]': 'tag',
    '[备注]': 'note'
}

def render_filename(template, roi):
    """根据模板和ROI数据渲染文件名。

    支持友好占位符如 '[索引]', '[标签]', '[X:03d]', '[面积:.1f]'。

    Args:
        template (str): 包含占位符的命名模板。
        roi (FrameROI): FrameROI对象。

    Returns:
        str: 渲染后的文件名。
    """
    rendered_name = template
    # Regex to find placeholders like [Name] or [Name:FormatSpec]
    placeholder_pattern = re.compile(r'\[([^\]:]+)(?::([^\]]+))?\]')

    for match in placeholder_pattern.finditer(template):
        placeholder = match.group(0) # e.g., '[索引:03d]'
        name_key = match.group(1)   # e.g., '索引'
        format_spec = match.group(2) # e.g., '03d' or None

        # Find the attribute name using the friendly key
        attr_name = PLACEHOLDER_MAP.get(f'[{name_key}]') 
        
        if attr_name and hasattr(roi, attr_name):
            value = getattr(roi, attr_name)
            try:
                # Apply format specifier if provided
                if format_spec:
                    formatted_value = f"{value:{format_spec}}"
                else:
                    # Default string conversion if no format spec
                    formatted_value = str(value)
                
                # Replace the placeholder in the rendered string
                rendered_name = rendered_name.replace(placeholder, formatted_value)
            except (ValueError, TypeError, Exception) as fmt_err:
                logging.warning(f"格式化占位符 '{placeholder}' 出错 (值: {value}, 格式: '{format_spec}'): {fmt_err}. 使用原始值替代。")
                # Fallback to string representation if format spec is invalid
                rendered_name = rendered_name.replace(placeholder, str(value))
        else:
            logging.warning(f"在模板中发现未知或无效的占位符: {placeholder}")

    # Basic filename sanitization (remove characters not suitable for filenames)
    # This is a simple example, might need refinement based on OS
    invalid_chars = r'[\\/:*?"<>|]'
    rendered_name = re.sub(invalid_chars, '_', rendered_name)
    
    # Ensure it ends with .png if no extension is specified
    if '.' not in os.path.splitext(rendered_name)[1]:
        rendered_name += ".png"
    elif not rendered_name.lower().endswith('.png'):
        # Handle cases where an incorrect extension might be present
        base, _ = os.path.splitext(rendered_name)
        rendered_name = base + ".png"

    return rendered_name

def sort_rois(rois, by="area", reverse=True):
    """根据指定属性对FrameROI列表进行排序。

    Args:
        rois (list[FrameROI]): FrameROI对象列表。
        by (str, optional): 排序依据的属性名称. Defaults to "area".
        reverse (bool, optional): 是否降序排序. Defaults to True.

    Returns:
        list[FrameROI]: 排序后的FrameROI对象列表。
    """
    if by == "idx": # 默认按提取顺序（idx升序）
        reverse = False
    return sorted(rois, key=lambda r: getattr(r, by), reverse=reverse)

def filter_rois(rois, area_range=None, aspect_range=None, x_range=None, y_range=None):
    """根据面积、长宽比、坐标范围筛选FrameROI列表。

    Args:
        rois (list[FrameROI]): FrameROI对象列表。
        area_range (tuple, optional): 面积范围 (min, max). Defaults to None.
        aspect_range (tuple, optional): 长宽比范围 (min, max). Defaults to None.
        x_range (tuple, optional): x坐标范围 (min, max). Defaults to None.
        y_range (tuple, optional): y坐标范围 (min, max). Defaults to None.

    Returns:
        list[FrameROI]: 筛选后的FrameROI对象列表。
    """
    result = []
    for r in rois:
        if area_range and not (area_range[0] <= r.area <= area_range[1]):
            continue
        if aspect_range and not (aspect_range[0] <= r.aspect_ratio <= aspect_range[1]):
            continue
        if x_range and not (x_range[0] <= r.x <= x_range[1]):
            continue
        if y_range and not (y_range[0] <= r.y <= y_range[1]):
            continue
        result.append(r)
    return result
