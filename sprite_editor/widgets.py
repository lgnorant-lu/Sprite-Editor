"""
---------------------------------------------------------------
File name:                  widgets.py
Author:                     Ignorant-lu
Date created:               2025/04/15
Description:                Sprite Mask Editor工具的自定义控件
----------------------------------------------------------------

Changed history:            
                            2025/04/15: 从sprite_mask_editor.py拆分为独立模块;
----
"""
import numpy as np
import cv2
from PIL import Image, ImageQt
from PyQt6 import QtWidgets, QtGui, QtCore
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QIcon, QAction, QCursor, QPixmap, QPainter, QPen, QColor
import logging

from .roi import FrameROI
from .constants import DEFAULT_BRUSH_SIZE

class ParamHelpLabel(QtWidgets.QLabel):
    """带悬浮帮助提示的QLabel。

    用于在UI中显示参数名称，并提供详细说明。
    """
    def __init__(self, text, help_text):
        """初始化ParamHelpLabel。

        Args:
            text (str): 标签显示的文本。
            help_text (str): 悬浮提示的帮助文本。
        """
        super().__init__(text)
        self.setToolTip(help_text)
        self.setStyleSheet("color:#333;font-weight:bold;")

class ThumbListWidget(QtWidgets.QScrollArea):
    """帧缩略图横向滚动浏览控件。

    显示提取的帧的缩略图列表，支持单选、多选（Ctrl/Shift）和右键菜单操作。

    Signals:
        frame_selected (int): 当用户选择单个帧时发出，参数为帧的索引。
        selection_changed (set): 当选择的帧集合发生变化时发出，参数为选中的索引集合。
        edit_mask_requested = QtCore.pyqtSignal() # Signal for edit mask request
        batch_export_requested = QtCore.pyqtSignal() # Signal for batch export request
        batch_tag_requested = QtCore.pyqtSignal() # Signal for batch tag request
        batch_note_requested = QtCore.pyqtSignal() # Signal for batch note request
    """
    frame_selected = QtCore.pyqtSignal(int)
    selection_changed = QtCore.pyqtSignal(set)
    # Define the custom signals
    edit_mask_requested = QtCore.pyqtSignal()
    batch_export_requested = QtCore.pyqtSignal()
    batch_tag_requested = QtCore.pyqtSignal()
    batch_note_requested = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        """初始化ThumbListWidget。

        Args:
            parent (QWidget, optional): 父控件. Defaults to None.
        """
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setMinimumHeight(130)
        self.setStyleSheet('background:#f7f9fb;border:none;')

        # 内容容器，用于放置缩略图标签
        self.thumb_container = QtWidgets.QWidget()
        self.setWidget(self.thumb_container)
        self.container_layout = QtWidgets.QHBoxLayout(self.thumb_container)
        self.container_layout.setContentsMargins(16, 14, 16, 14)
        self.container_layout.setSpacing(12)
        self.container_layout.addStretch() # 添加伸缩项以保持缩略图左对齐

        # 内部状态
        self.thumbs = [] # 存储原始图像数据 (numpy arrays)
        self.current_idx = -1 # 当前选中的单个帧索引
        self.selected_indices = set() # 当前选中的所有帧索引集合
        self.thumb_labels = [] # 存储QLabel控件

    def set_thumbs(self, imgs):
        """设置并显示缩略图列表。

        会先清空现有的缩略图。

        Args:
            imgs (list[np.ndarray]): 包含帧图像数据(RGBA)的列表。
        """
        # 清空现有缩略图
        self.clear_thumbs()
        self.thumbs = imgs

        # 为每个图像创建并添加QLabel缩略图
        for i, img in enumerate(self.thumbs):
            label = QtWidgets.QLabel()
            label.setFixedSize(90, 90)
            label.setStyleSheet("background:#fff;border-radius:6px;padding:5px;")

            # 转换为QPixmap并缩放
            try:
                qimg = ImageQt.ImageQt(Image.fromarray(img))
                thumb = QtGui.QPixmap.fromImage(qimg)
                thumb = thumb.scaled(80, 80, Qt.AspectRatioMode.KeepAspectRatio,
                                    Qt.TransformationMode.SmoothTransformation)
                label.setPixmap(thumb)
            except Exception as e:
                # 处理图像转换或缩放错误
                logging.error(f"Error creating thumbnail for index {i}: {e}")
                label.setText("错误") # 显示错误提示

            label.setAlignment(Qt.AlignmentFlag.AlignCenter)

            # 将标签添加到布局的末尾（在伸缩项之前）
            self.container_layout.insertWidget(self.container_layout.count()-1, label)
            self.thumb_labels.append(label)

        # 如果有缩略图，默认选中第一个
        if self.thumbs:
            self.set_current(0)
            self.selected_indices = {0}
            self.selection_changed.emit(self.selected_indices)
            self.update_selection_visuals()

    def clear_thumbs(self):
        """清空所有缩略图和内部状态。"""
        self.thumbs = []
        self.current_idx = -1
        self.selected_indices = set()

        # 从布局中移除并删除所有QLabel控件
        for label in self.thumb_labels:
            self.container_layout.removeWidget(label)
            label.deleteLater()
        self.thumb_labels = []

    def set_current(self, idx):
        """设置当前选中的单个帧。

        Args:
            idx (int): 要选中的帧的索引。
        """
        if 0 <= idx < len(self.thumbs):
            # 如果点击的已经是当前帧，则不重复触发信号，但确保视觉更新
            is_new_selection = (self.current_idx != idx)
            self.current_idx = idx
            if is_new_selection:
                self.frame_selected.emit(idx)
            # 更新视觉效果，确保边框正确
            self.update_selection_visuals()

    def select_frames(self, indices):
        """设置当前选择的帧集合。

        Args:
            indices (set[int]): 要选择的帧的索引集合。
        """
        new_selection = set(filter(lambda i: 0 <= i < len(self.thumbs), indices))
        if new_selection != self.selected_indices:
            self.selected_indices = new_selection
            self.selection_changed.emit(self.selected_indices)
            self.update_selection_visuals()

    def update_selection_visuals(self):
        """根据当前选中状态更新所有缩略图的边框样式。"""
        for i, label in enumerate(self.thumb_labels):
            if i == self.current_idx:
                # 当前帧: 粗蓝色边框
                label.setStyleSheet("background:#fff;border:3px solid #4f8cff;border-radius:6px;padding:2px;")
            elif i in self.selected_indices:
                # 多选中的其他帧: 细浅蓝色边框
                label.setStyleSheet("background:#fff;border:2px solid #a0c8ff;border-radius:6px;padding:3px;")
            else:
                # 未选中的帧: 无边框
                label.setStyleSheet("background:#fff;border:none;border-radius:6px;padding:5px;")

    def mousePressEvent(self, e: QtGui.QMouseEvent):
        """处理鼠标点击事件，实现单选、Ctrl多选、Shift范围选择。"""
        super().mousePressEvent(e)
        # 将事件坐标转换为内容容器的坐标
        widget_pos = self.thumb_container.mapFromGlobal(e.globalPosition().toPoint())
        if self.thumb_container.rect().contains(widget_pos):
            clicked_idx = -1
            # 查找被点击的缩略图标签
            for i, label in enumerate(self.thumb_labels):
                if label.geometry().contains(widget_pos):
                    clicked_idx = i
                    break

            if clicked_idx != -1:
                modifiers = e.modifiers()
                if modifiers & Qt.KeyboardModifier.ControlModifier:
                    # Ctrl + 点击: 切换选中状态
                    current_selection = self.selected_indices.copy()
                    if clicked_idx in current_selection:
                        current_selection.remove(clicked_idx)
                    else:
                        current_selection.add(clicked_idx)
                    self.select_frames(current_selection)
                    # 如果Ctrl点击的是当前帧，更新当前帧为最近添加或移除的帧，
                    # 或者保持不变（这里选择保持current_idx不变，因为Ctrl主要用于集合操作）

                elif modifiers & Qt.KeyboardModifier.ShiftModifier and self.current_idx >= 0:
                    # Shift + 点击: 范围选择 (从 current_idx 到 clicked_idx)
                    start_idx = min(clicked_idx, self.current_idx)
                    end_idx = max(clicked_idx, self.current_idx)
                    range_selection = set(range(start_idx, end_idx + 1))
                    self.select_frames(range_selection)
                    # Shift选择后，将被点击的帧设为当前帧
                    self.set_current(clicked_idx)

                else:
                    # 普通点击: 选择单个帧，并设为当前帧
                    self.set_current(clicked_idx)
                    self.select_frames({clicked_idx}) # 更新选择集为单项

                # 确保视觉效果总是更新
                self.update_selection_visuals()

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent):
        """处理右键菜单事件。"""
        widget_pos = self.thumb_container.mapFromGlobal(event.globalPos())
        clicked_idx = -1
        for i, label in enumerate(self.thumb_labels):
            if label.geometry().contains(widget_pos):
                clicked_idx = i
                break

        if clicked_idx != -1 and clicked_idx not in self.selected_indices:
            self.set_current(clicked_idx)
            self.select_frames({clicked_idx})
            self.update_selection_visuals()

        if self.selected_indices:
            menu = QtWidgets.QMenu(self)

            if len(self.selected_indices) == 1 and self.current_idx != -1:
                edit_mask_action = QAction(QIcon(":/icons/edit.png"), "编辑Mask", self)
                # Connect action to custom signal emit
                edit_mask_action.triggered.connect(self.edit_mask_requested.emit) 
                menu.addAction(edit_mask_action)
                menu.addSeparator()

            export_action = QAction(QIcon(":/icons/batch_export.png"), f"导出选中帧 ({len(self.selected_indices)})", self)
            set_tag_action = QAction(QIcon(":/icons/batch_tag.png"), "批量设置标签", self)
            set_note_action = QAction(QIcon(":/icons/batch_note.png"), "批量设置备注", self)

            # Connect actions to custom signal emit
            export_action.triggered.connect(self.batch_export_requested.emit)
            set_tag_action.triggered.connect(self.batch_tag_requested.emit)
            set_note_action.triggered.connect(self.batch_note_requested.emit)

            menu.addAction(export_action)
            menu.addAction(set_tag_action)
            menu.addAction(set_note_action)

            menu.exec(event.globalPos())

    def enterEvent(self, event):
        """鼠标进入控件区域时，设置手型光标。"""
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        super().enterEvent(event)

    def leaveEvent(self, event):
        """鼠标离开控件区域时，恢复箭头光标。"""
        self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        super().leaveEvent(event)

class MaskEditWidget(QtWidgets.QWidget):
    """用于手绘编辑蒙版的可视化控件。

    继承自QLabel，显示带有蒙版效果的图像。
    支持使用画笔（添加蒙版）和橡皮（移除蒙版）进行绘制。
    提供撤销、重做、重置和自动修补功能。
    支持缩放和平移功能。

    Signals:
        mask_edited (np.ndarray): 当蒙版被编辑（绘制操作释放鼠标时）发出。
        grabcut_mode_changed (bool): Signal to request showing/hiding GrabCut buttons in the parent dialog
    """
    mask_edited = QtCore.pyqtSignal(np.ndarray)
    grabcut_mode_changed = QtCore.pyqtSignal(bool)

    def __init__(self, base_img, mask, parent_dialog, parent=None):
        """初始化MaskEditWidget。

        Args:
            base_img (np.ndarray): 基础图像数据 (RGBA)。
            mask (np.ndarray): 初始蒙版数据 (灰度图, H, W)，尺寸应与base_img匹配。
            parent_dialog (QWidget): Reference to the parent dialog for GC button toggling
            parent (QWidget, optional): 父控件. Defaults to None.
        """
        super().__init__(parent)
        # 确保传入的mask和图像尺寸一致
        if base_img.shape[:2] != mask.shape[:2]:
             # 如果不一致，优先调整mask尺寸以匹配图像
             logging.warning(f"MaskEditWidget received mask with shape {mask.shape[:2]} but image shape is {base_img.shape[:2]}. Resizing mask.")
             self.mask = cv2.resize(mask, (base_img.shape[1], base_img.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            self.mask = mask.copy()
             
        self.base_img = base_img.copy() # 存储基础图像
        self.drawing = False # 标记是否正在绘制
        self.brush_size = DEFAULT_BRUSH_SIZE # 画笔大小
        self.mode = 'draw' # 当前模式: 'draw' 或 'erase'
        self.last_point = None # 上一个绘制点

        # GrabCut related state
        self.grabcut_mode = None # None, 'rect', 'refine'
        self.refine_mode = 'fg' # 'fg' or 'bg' for refinement drawing
        self.grabcut_rect = None
        self.drawing_rect = False
        self.rect_start_point = None
        self.gc_mask = None # Internal grabCut mask (stores 0,1,2,3)
        self.gc_bgd_model = None
        self.gc_fgd_model = None
        self.gc_initialized = False

        # Watershed related state
        self.watershed_markers = None
        self.watershed_marker_mode = 'fg' # 'fg', 'bg'

        # Store parent dialog reference to toggle buttons
        self.parent_dialog = parent_dialog

        # 缩放和平移相关
        self.zoom_factor = 1.0 # 缩放系数
        self.pan_offset = QtCore.QPoint(0, 0) # 平移偏移量
        self.panning = False # 是否正在平移
        self.last_pan_pos = None # 上一次平移位置
        
        # 控件基本设置
        self.setMinimumSize(384, 384) # 设置最小尺寸
        self.setCursor(QCursor(Qt.CursorShape.CrossCursor)) # 设置十字光标

        # 编辑历史记录
        self.history = [self.mask.copy()] # 初始状态加入历史
        self.history_idx = 0 # 当前历史指针

        self._pixmap = None  # 用于缓存当前显示的 QPixmap
        self.update_pix()
        
        # 启用鼠标追踪
        self.setMouseTracking(True)
        
        # 启用接收鼠标滚轮事件
        self.setFocusPolicy(Qt.FocusPolicy.WheelFocus)

        # 启用接收键盘事件
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        
        # 创建快捷键映射表
        self.shortcuts = {
            QtCore.Qt.Key.Key_Z: self.undo,                  # Z - 撤销
            QtCore.Qt.Key.Key_Y: self.redo,                  # Y - 重做
            QtCore.Qt.Key.Key_D: lambda: self.set_mode('draw'),  # D - 设置为绘制模式
            QtCore.Qt.Key.Key_E: lambda: self.set_mode('erase'), # E - 设置为擦除模式
            QtCore.Qt.Key.Key_R: self.reset_view,            # R - 重置视图
            QtCore.Qt.Key.Key_C: self.clear_mask,            # C - 清除蒙版
            QtCore.Qt.Key.Key_Equal: self.increase_brush_size,    # + - 增加画笔大小
            QtCore.Qt.Key.Key_Minus: self.decrease_brush_size,    # - - 减小画笔大小
            QtCore.Qt.Key.Key_0: lambda: self.reset_zoom(),       # 0 - 重置缩放
        }

    def set_brush_size(self, size):
        """设置画笔/橡皮大小。

        Args:
            size (int): 新的画笔大小。
        """
        self.brush_size = size

    def set_mode(self, mode):
        """设置编辑模式。 Handles standard modes and GrabCut initiation.

        Args:
            mode (str): 'draw', 'erase', 'grabcut_rect', 'grabcut_refine', 'watershed_mark'.
        """
        if mode == self.mode:
            return
            
        self.mode = mode
        self.drawing = False # Reset drawing state when changing mode
        self.drawing_rect = False
        self.panning = False

        if mode == 'grabcut_refine':
            self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor)) # Example cursor for refine
            self.grabcut_mode_changed.emit(True) # Request showing GC buttons
            # Default to FG marker when entering refine mode
            self.refine_mode = 'fg' 
            # Ensure FG button is checked in dialog (dialog should handle this on show)
            # self.parent_dialog.btn_gc_fg.setChecked(True)
        elif mode == 'grabcut_rect':
            self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
            self.grabcut_mode_changed.emit(False) # Hide GC buttons during rect draw
        elif mode == 'watershed_mark':
            self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor)) # Or custom marker cursor
            self.grabcut_mode_changed.emit(False) # Hide GC buttons
            # Need signal/call to show WS buttons
            if hasattr(self.parent_dialog, 'show_watershed_buttons'):
                self.parent_dialog.show_watershed_buttons(True)
            else: 
                logging.warning("Parent dialog does not have show_watershed_buttons method")
            self.gc_initialized = False # Exit GrabCut session
            self.gc_mask = None
            # Initialize markers when entering mode
            if self.mask is not None:
                 self.watershed_markers = np.zeros(self.mask.shape[:2], dtype=np.int32)
            else: 
                 self.watershed_markers = None # Cannot init without mask size
        else: # draw, erase, or other standard modes
            self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
            self.grabcut_mode_changed.emit(False) # Hide GC buttons
            self.gc_initialized = False # Exit GrabCut session
            self.gc_mask = None
            # Need signal/call to hide WS buttons
            if hasattr(self.parent_dialog, 'show_watershed_buttons'):
                self.parent_dialog.show_watershed_buttons(False)
            self.watershed_markers = None # Clear markers when exiting mode

    def enter_grabcut_rect_mode(self):
        """进入 GrabCut 矩形框选模式。"""
        logging.info("Entering GrabCut rectangle selection mode.")
        # Set mode directly instead of calling self.set_mode to avoid recursive signals/state changes
        self.mode = 'grabcut_rect' 
        self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
        self.grabcut_rect = None
        self.drawing_rect = False
        self.rect_start_point = None
        self.gc_initialized = False # Ensure GC is not considered initialized yet
        self.gc_mask = None
        self.grabcut_mode_changed.emit(False) # Hide GC buttons
        self.update() # Update cursor and allow paintEvent to draw hints if needed

    def set_grabcut_marker_mode(self, marker_mode):
        """设置 GrabCut 修正模式下的标记类型（前景或背景）。"""
        if self.mode != 'grabcut_refine':
            # If not already in refine mode, enter it first
            self.set_mode('grabcut_refine')
            
        if marker_mode in ['fg', 'bg']:
            self.refine_mode = marker_mode
            logging.info(f"Set GrabCut marker mode to: {marker_mode}")
            # Optionally change cursor based on fg/bg
            # self.setCursor(...) 
        else:
            logging.warning(f"Invalid marker mode: {marker_mode}")

    def enter_watershed_mode(self):
        """进入 Watershed 标记模式。"""
        logging.info("Entering Watershed marker mode.")
        self.set_mode('watershed_mark') # Let set_mode handle common logic
        # Initial marker mode is set within set_mode if needed
        # Initialization of self.watershed_markers also happens in set_mode
        if self.watershed_markers is None and self.mask is not None:
             # Fallback initialization if mask was loaded after mode change attempt
             self.watershed_markers = np.zeros(self.mask.shape[:2], dtype=np.int32)
             logging.warning("Initialized watershed markers late.")
        elif self.mask is None:
             logging.error("Cannot enter watershed mode properly without a base mask/image.")
             # Should potentially revert mode back to draw?
             self.set_mode('draw') 
             QtWidgets.QMessageBox.critical(self.parent_dialog, "错误", "无法初始化分水岭模式，请确保图像已加载。")
             return
        # Set default marker mode button checked in dialog
        if hasattr(self.parent_dialog, 'btn_ws_fg'):
            self.parent_dialog.btn_ws_fg.setChecked(True)
        self.update() # Ensure visualization updates

    def set_watershed_marker_mode(self, mode):
        """设置 Watershed 标记模式 ('fg' 或 'bg')。"""
        logging.info(f"Setting Watershed marker mode to: {mode}")
        if self.mode != 'watershed_mark':
            logging.info(f"Current mode is not watershed_mark, switching from {self.mode}")
            self.set_mode('watershed_mark')
        
        if mode in ['fg', 'bg']:
            self.watershed_marker_mode = mode
            logging.info(f"Watershed marker mode is now: {mode}")
            # Optionally change cursor
        else:
            logging.warning(f"Invalid Watershed marker mode: {mode}")

    def finish_grabcut(self):
        """完成 GrabCut 分割，应用最终蒙版并返回绘制模式。"""
        if not self.gc_initialized or self.gc_mask is None:
            logging.warning("GrabCut not initialized, cannot finish.")
            # Force back to draw mode anyway
            self.set_mode('draw')
            return

        logging.info("Finishing GrabCut session.")
        try:
            # Generate final binary mask from the internal gc_mask
            output_mask = np.where((self.gc_mask == cv2.GC_FGD) | (self.gc_mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')
            # Use set_mask WITHOUT resetting grabcut state here, 
            # as we are finalizing *this* session.
            # self.set_mask will be called internally by push_history if needed,
            # or we can update self.mask directly and call push_history.
            
            # Directly update mask and history
            self.mask = output_mask
            self.push_history() # Add final mask to history
            self.update_pix() # Update view with final binary mask
            self.mask_edited.emit(self.mask)

        except Exception as e:
            logging.exception(f"Error finalizing GrabCut mask: {e}")
        finally:
            # Reset all GrabCut state and return to draw mode
            self.gc_initialized = False
            self.gc_mask = None
            self.gc_bgd_model = None
            self.gc_fgd_model = None
            self.grabcut_rect = None
            self.drawing_rect = False
            self.rect_start_point = None
            self.grabcut_mode = None
            self.set_mode('draw') # This will also hide GC buttons via signal
            
    def _draw_grabcut_marker(self, point):
        """在 GrabCut 内部掩码上绘制单个标记点。"""
        if point is None or self.gc_mask is None or not self.gc_initialized:
            return
            
        x, y = point
        marker_value = cv2.GC_FGD if self.refine_mode == 'fg' else cv2.GC_BGD
        # Use a slightly larger marker for visibility? Adapt brush size?
        marker_size = max(1, self.brush_size // 2)
        cv2.circle(self.gc_mask, (x, y), marker_size, marker_value, -1, cv2.LINE_AA)
        # self.update() # Optional: update display to show markers live
        # Need a way to visualize markers if update() is called here, maybe in paintEvent

    def _draw_grabcut_line(self, p1, p2):
        """在 GrabCut 内部掩码上绘制标记线条。"""
        if p1 is None or p2 is None or self.gc_mask is None or not self.gc_initialized:
            return
            
        x1, y1 = p1
        x2, y2 = p2
        marker_value = cv2.GC_FGD if self.refine_mode == 'fg' else cv2.GC_BGD
        cv2.line(self.gc_mask, (x1, y1), (x2, y2), marker_value, self.brush_size, cv2.LINE_AA)
        # self.update() # Optional: update display to show markers live
        # Need a way to visualize markers if update() is called here

    def _draw_watershed_marker(self, point):
        """在 Watershed 标记图上绘制单个标记点。"""
        logging.debug(f"_draw_watershed_marker called with point: {point}")
        if point is None or self.watershed_markers is None:
            logging.debug(f"Cannot draw watershed marker: point is None? {point is None}, watershed_markers is None? {self.watershed_markers is None}")
            return
        x, y = point
        marker_value = 1 if self.watershed_marker_mode == 'fg' else 2 # 1 for FG, 2 for BG
        marker_size = max(1, self.brush_size // 2)
        logging.debug(f"Drawing watershed marker at ({x}, {y}) with value {marker_value} and size {marker_size}")
        cv2.circle(self.watershed_markers, (x, y), marker_size, marker_value, -1)
        
        # 添加调试代码检查标记是否被写入
        marker_count_before = np.sum(self.watershed_markers == marker_value)
        temp_debug_markers = np.zeros_like(self.watershed_markers)
        cv2.circle(temp_debug_markers, (x, y), marker_size, marker_value, -1)
        expected_new_markers = np.sum(temp_debug_markers == marker_value)
        marker_count_after = np.sum(self.watershed_markers == marker_value)
        logging.debug(f"Marker count before: {marker_count_before}, expected new: {expected_new_markers}, after: {marker_count_after}")
        # Remove update from here, rely on caller (mousePress/Move) to update
        # self.update() 

    def _draw_watershed_marker_line(self, p1, p2):
        """在 Watershed 标记图上绘制标记线条。"""
        if p1 is None or p2 is None or self.watershed_markers is None:
            return
        x1, y1 = p1
        x2, y2 = p2
        marker_value = 1 if self.watershed_marker_mode == 'fg' else 2 # 1 for FG, 2 for BG
        cv2.line(self.watershed_markers, (x1, y1), (x2, y2), marker_value, self.brush_size)
        # Remove update from here, rely on caller (mouseMove) to update
        # self.update() 

    def mousePressEvent(self, e: QtGui.QMouseEvent):
        """处理鼠标按下事件，开始绘制、平移或框选。"""
        if e.button() == Qt.MouseButton.LeftButton:
            modifiers = QtWidgets.QApplication.keyboardModifiers()
            if self.mode == 'grabcut_rect':
                self.drawing_rect = True
                self.rect_start_point = e.pos()
                self.grabcut_rect = QtCore.QRect(self.rect_start_point, e.pos()) # Initialize rect
                self.update()
            elif modifiers == Qt.KeyboardModifier.ShiftModifier or e.modifiers() == Qt.KeyboardModifier.ShiftModifier:
                self.panning = True
                self.last_pan_pos = e.pos()
                self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
            elif self.mode in ['draw', 'erase']:
                self.drawing = True
                self.last_point = e.pos()
                img_point = self.widget_to_image_coords(self.last_point)
                if img_point is not None:
                    self.draw_point(img_point)
            elif self.mode == 'grabcut_refine':
                self.drawing = True
                self.last_point = e.pos()
                img_point = self.widget_to_image_coords(self.last_point)
                if img_point is not None:
                    self._draw_grabcut_marker(img_point)
            elif self.mode == 'watershed_mark':
                self.drawing = True
                self.last_point = e.pos()
                img_point = self.widget_to_image_coords(self.last_point)
                if img_point is not None:
                    self._draw_watershed_marker(img_point)
                self.update_pix() # Ensure pixmap is recalculated after drawing the first point
            # Handle other modes if added later

    def mouseMoveEvent(self, e: QtGui.QMouseEvent):
        """处理鼠标移动事件，进行绘制、平移或更新矩形/标记。"""
        if self.drawing_rect:
            self.grabcut_rect = QtCore.QRect(self.rect_start_point, e.pos()).normalized()
            self.update()
        elif self.panning:
            delta = e.pos() - self.last_pan_pos
            self.pan_offset += delta
            self.last_pan_pos = e.pos()
            self.update_pix()
        elif self.drawing:
            current_pos = e.pos()
            img_point1 = self.widget_to_image_coords(self.last_point)
            img_point2 = self.widget_to_image_coords(current_pos)
            if img_point1 is not None and img_point2 is not None:
                if self.mode in ['draw', 'erase']:
                    self.draw_line_on_mask(img_point1, img_point2)
                elif self.mode == 'grabcut_refine':
                    self._draw_grabcut_line(img_point1, img_point2)
                elif self.mode == 'watershed_mark':
                     self._draw_watershed_marker_line(img_point1, img_point2)
                     # Update is now called reliably after the drawing logic
                     self.update_pix() # Recalculate pixmap for continuous drawing
            self.last_point = current_pos
            # Remove the potentially redundant update from here
            # if self.mode == 'watershed_mark': self.update() 

    def mouseReleaseEvent(self, e: QtGui.QMouseEvent):
        """处理鼠标释放事件，结束绘制、平移或执行 GrabCut 初始化/细化。"""
        if e.button() == Qt.MouseButton.LeftButton:
            if self.drawing_rect:
                self.drawing_rect = False
                if self.grabcut_rect and self.grabcut_rect.width() > 5 and self.grabcut_rect.height() > 5: # Min size check
                    logging.info(f"GrabCut rectangle defined: {self.grabcut_rect}")
                    self._run_grabcut_initial() # Call the initial GrabCut logic
                else:
                    logging.warning("GrabCut rectangle too small or invalid.")
                    self.grabcut_rect = None # Reset invalid rect
                self.rect_start_point = None
                # Maybe switch back to default mode or to refine mode here?
                # For now, stay in rect mode until GrabCut logic decides
                self.update() # Redraw without the rect
            elif self.panning:
                self.panning = False
                # Change cursor back based on current mode
                if self.mode == 'grabcut_rect':
                    self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
                else: # Default or draw/erase
                    self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
            elif self.drawing:
                if self.mode in ['draw', 'erase']:
                    self.drawing = False
                    self.last_point = None
                    self.push_history()
                    self.mask_edited.emit(self.mask)
                elif self.mode == 'grabcut_refine':
                    self.drawing = False
                    self.last_point = None
                    self._run_grabcut_refine() # Run refinement iteration
                elif self.mode == 'watershed_mark':
                     self.drawing = False
                     self.last_point = None
                     # Don't run segmentation on release, wait for button press

    def wheelEvent(self, e: QtGui.QWheelEvent):
        """处理鼠标滚轮事件，缩放图像。"""
        # 计算鼠标位置相对于控件的比例
        mouse_pos = e.position()
        widget_rect = self.rect()
        rel_x = mouse_pos.x() / widget_rect.width()
        rel_y = mouse_pos.y() / widget_rect.height()
        
        # 获取滚轮前进的角度，正值表示向前滚动(放大)，负值表示向后滚动(缩小)
        degrees = e.angleDelta().y() / 8
        steps = degrees / 15  # 一个标准步长是15度
        
        # 缩放系数变化: 每次改变15%
        zoom_delta = 0.15 * steps
        old_zoom = self.zoom_factor
        
        # 更新缩放系数，限制在合理范围内
        self.zoom_factor = max(0.1, min(10.0, self.zoom_factor + zoom_delta))
        
        # 调整平移偏移以保持鼠标所指位置
        if old_zoom != self.zoom_factor:
            # 计算新旧比例
            zoom_ratio = self.zoom_factor / old_zoom
            
            # 计算相对偏移中心
            center_x = widget_rect.width() * rel_x
            center_y = widget_rect.height() * rel_y
            
            # 调整偏移，使鼠标位置固定
            self.pan_offset.setX(int((self.pan_offset.x() - center_x) * zoom_ratio + center_x))
            self.pan_offset.setY(int((self.pan_offset.y() - center_y) * zoom_ratio + center_y))
            
            # 更新显示
            self.update_pix()

    def widget_to_image_coords(self, p: QtCore.QPoint):
        """
        将控件坐标转换为图像坐标（考虑缩放和平移）
        
        Args:
            p (QPoint): 控件坐标
        
        Returns:
            tuple: 图像坐标 (x, y)
        """
        if self._pixmap is None or self.mask is None: # Also check self.mask for shape info
            return None
            
        img_h, img_w = self.mask.shape[:2]
        if img_w <= 0 or img_h <= 0: return None # Avoid division by zero
        
        widget_w, widget_h = self.width(), self.height()
        
        # Calculate the size of the pixmap as currently displayed (considering zoom)
        # We need the original pixmap size before scaling to fit the widget initially
        # Let's use the actual base image dimensions for calculation
        base_img_h, base_img_w = self.base_img.shape[:2]
        
        display_w = base_img_w * self.zoom_factor
        display_h = base_img_h * self.zoom_factor
        
        # Calculate the top-left corner of the displayed image within the widget (considering pan)
        # This needs careful re-evaluation based on how update_pix draws the pixmap
        # Assuming update_pix centers the potentially scaled pixmap and then applies pan:
        current_pixmap_w = base_img_w * self.zoom_factor
        current_pixmap_h = base_img_h * self.zoom_factor
        
        # Top-left corner if centered without pan
        centered_x0 = (widget_w - current_pixmap_w) / 2
        centered_y0 = (widget_h - current_pixmap_h) / 2
        
        # Add pan offset
        img_x0_widget = centered_x0 + self.pan_offset.x()
        img_y0_widget = centered_y0 + self.pan_offset.y()
        
        # Calculate scaling factor from widget display size to original image size
        scale_w = base_img_w / current_pixmap_w if current_pixmap_w > 0 else 1
        scale_h = base_img_h / current_pixmap_h if current_pixmap_h > 0 else 1
        
        # Convert widget point to image point
        img_x = (p.x() - img_x0_widget) * scale_w
        img_y = (p.y() - img_y0_widget) * scale_h
        
        # Clamp coordinates to image boundaries [0, width-1] and [0, height-1]
        img_x_clamped = int(max(0, min(base_img_w - 1, img_x)))
        img_y_clamped = int(max(0, min(base_img_h - 1, img_y)))
            
        return (img_x_clamped, img_y_clamped)

    def _rect_widget_to_image(self, rect: QtCore.QRect):
        """将控件坐标的矩形转换为图像坐标的矩形。"""
        if rect is None:
            return None
            
        # Convert top-left and bottom-right points
        top_left_img = self.widget_to_image_coords(rect.topLeft())
        bottom_right_img = self.widget_to_image_coords(rect.bottomRight())

        if top_left_img is not None and bottom_right_img is not None:
            x1, y1 = top_left_img
            x2, y2 = bottom_right_img
            # Ensure x1 < x2 and y1 < y2
            img_x = min(x1, x2)
            img_y = min(y1, y2)
            img_w = abs(x1 - x2)
            img_h = abs(y1 - y2)
            # Clamp to image boundaries (important for grabCut)
            img_h_orig, img_w_orig = self.mask.shape[:2]
            img_x = max(0, img_x)
            img_y = max(0, img_y)
            img_w = min(img_w_orig - img_x, img_w)
            img_h = min(img_h_orig - img_y, img_h)
            
            # Add check for valid width and height
            if img_w > 0 and img_h > 0:
                return (img_x, img_y, img_w, img_h)
            else:
                logging.warning(f"Calculated image rectangle has non-positive dimensions: w={img_w}, h={img_h}")
        
        logging.warning("Could not convert widget rectangle to valid image rectangle.")
        return None

    def _run_grabcut_initial(self):
        """执行初始 GrabCut 算法并进入细化模式。"""
        if self.grabcut_rect is None or self.base_img is None:
            logging.warning("GrabCut rectangle or base image not available.")
            return

        logging.info("Running initial GrabCut...")
        
        # Convert widget coordinates rect to image coordinates rect
        img_rect = self._rect_widget_to_image(self.grabcut_rect)
        if img_rect is None:
            logging.error("Failed to convert rectangle to image coordinates for GrabCut.")
            self.grabcut_rect = None
            self.mode = 'draw'
            self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
            self.update()
            return

        try:
            # Ensure image is BGR for grabCut
            if self.base_img.shape[2] == 4:
                img_bgr = cv2.cvtColor(self.base_img[:, :, :3], cv2.COLOR_RGBA2BGR)
            else:
                img_bgr = self.base_img[:, :, :3].copy() # Assume RGB, convert to BGR copy
                img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)
            
            # Initialize mask and models for grabCut
            self.gc_mask = np.zeros(img_bgr.shape[:2], np.uint8)
            self.gc_bgd_model = np.zeros((1, 65), np.float64)
            self.gc_fgd_model = np.zeros((1, 65), np.float64)
            
            # Run grabCut with rect initialization
            iter_count = 5 # Initial iterations
            logging.info(f"Running cv2.grabCut with GC_INIT_WITH_RECT, rect={img_rect}, iters={iter_count}")
            cv2.grabCut(img_bgr, self.gc_mask, img_rect, self.gc_bgd_model, self.gc_fgd_model, iter_count, cv2.GC_INIT_WITH_RECT)
            self.gc_initialized = True
            
            # Don't convert to binary mask yet, update pixmap for refine mode
            # self.set_mask(...) REMOVED
            logging.info("Initial GrabCut finished. Entering refine mode.")
            self.update_pix_for_grabcut() # Show initial PR_FG/PR_BG results
            self.set_mode('grabcut_refine') # Enter refine mode
            # self.set_grabcut_marker_mode('fg') # set_mode already defaults to fg

        except Exception as e:
            logging.exception(f"Error during initial GrabCut execution: {e}")
            # Reset state and return to draw mode on error
            self.finish_grabcut() # Call finish to properly reset everything
        # Removed finally block here, state handled by set_mode or finish_grabcut

    def _run_grabcut_refine(self):
        """执行 GrabCut 细化迭代。"""
        if not self.gc_initialized or self.gc_mask is None or self.base_img is None:
            logging.warning("GrabCut not initialized or mask/image missing for refinement.")
            return

        logging.info("Running GrabCut refinement iteration...")
        try:
            # Ensure image is BGR
            if self.base_img.shape[2] == 4:
                img_bgr = cv2.cvtColor(self.base_img[:, :, :3], cv2.COLOR_RGBA2BGR)
            else:
                img_bgr = cv2.cvtColor(self.base_img[:, :, :3].copy(), cv2.COLOR_RGB2BGR)
                
            # Run grabCut evaluation using the current gc_mask with user scribbles
            iter_count = 1 # Usually 1 iteration is enough for refinement
            cv2.grabCut(img_bgr, self.gc_mask, None, self.gc_bgd_model, self.gc_fgd_model, iter_count, cv2.GC_EVAL)
            
            # Update the visual representation based on the new gc_mask
            self.update_pix_for_grabcut()
            logging.info("GrabCut refinement iteration finished.")
            
            # Generate the temporary binary mask for external use (like history preview maybe?)
            # output_mask = np.where((self.gc_mask == cv2.GC_FGD) | (self.gc_mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')
            # self.mask = output_mask # Don't update self.mask until finish_grabcut
            # Don't push to history on each refine step
            # self.mask_edited.emit(self.mask)

        except Exception as e:
            logging.exception(f"Error during GrabCut refinement: {e}")
            # Maybe reset to state before this refine step?
            # Or just let user try again or finish.

    def draw_point(self, point):
        """在mask上绘制一个点

        Args:
            point (tuple): 图像坐标点(x, y)
        """
        if point is None:
            return
            
        x, y = point
        color = 255 if self.mode == 'draw' else 0
        cv2.circle(self.mask, (x, y), self.brush_size // 2, color, -1, cv2.LINE_AA)
        self.update_pix()

    def draw_line_on_mask(self, p1, p2):
        """在蒙版上绘制一条线（画笔或橡皮）。

        Args:
            p1 (tuple): 线条起点(x, y)
            p2 (tuple): 线条终点(x, y)
        """
        if p1 is None or p2 is None:
            return
            
        # 获取图像坐标
        x1, y1 = p1
        x2, y2 = p2
        
        # 根据模式确定绘制颜色 (画笔=255, 橡皮=0)
        color = 255 if self.mode == 'draw' else 0
        
        # 使用cv2.line在蒙版上绘制
        cv2.line(self.mask, (x1, y1), (x2, y2), color, self.brush_size, cv2.LINE_AA)
        
        # 更新显示
        self.update_pix()

    def update_pix(self):
        """更新控件显示的Pixmap。如果正在 GrabCut/Watershed 细化/标记，则调用专用可视化。"""
        logging.debug(f"update_pix called with mode: {self.mode}")
        logging.debug(f"Mode type: {type(self.mode)}, Value: '{self.mode}'")
        mode_equals_watershed = self.mode == 'watershed_mark'
        logging.debug(f"Mode equals 'watershed_mark': {mode_equals_watershed}")
        
        if self.mode == 'grabcut_refine' and self.gc_initialized:
             # Use the specialized visualization during refinement
             logging.debug("Calling update_pix_for_grabcut")
             self.update_pix_for_grabcut()
             return
        elif self.mode == 'watershed_mark':
            logging.debug("Calling update_pix_for_watershed")
            self.update_pix_for_watershed()
            return
        # Standard visualization for draw/erase/etc.
        if self.base_img is None or self.mask is None:
            self.clear()
            return
        overlay = self.base_img.copy()
        img_h, img_w = overlay.shape[:2]
        if self.mask.shape[:2] != (img_h, img_w):
            # 如果尺寸不匹配，可能是在初始化时调整过，或者外部直接设置了不同尺寸的mask
            # 这里再次调整以确保一致性
            logging.warning(f"update_pix resizing mask from {self.mask.shape[:2]} to {(img_h, img_w)}")
            mask_resized = cv2.resize(self.mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        else:
            mask_resized = self.mask
        mask_bg_bool = (mask_resized == 0)
        overlay_with_effect = overlay.copy()
        overlay_with_effect[mask_bg_bool] = [120, 120, 120, 128]
        try:
            qimg = ImageQt.ImageQt(Image.fromarray(overlay_with_effect))
            pixmap = QPixmap.fromImage(qimg)
            # 缩放和平移 - 统一处理方式，不再区分是否有缩放/平移
            transform = QtGui.QTransform()
            transform.scale(self.zoom_factor, self.zoom_factor)
            pixmap = pixmap.transformed(transform, Qt.TransformationMode.SmoothTransformation)
            result_pixmap = QPixmap(self.size())
            result_pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(result_pixmap)
            x = (self.width() - pixmap.width()) / 2 + self.pan_offset.x()
            y = (self.height() - pixmap.height()) / 2 + self.pan_offset.y()
            painter.drawPixmap(int(x), int(y), pixmap)
            painter.end()
            self._pixmap = result_pixmap
        except Exception as e:
            logging.exception(f"Error updating pixmap: {e}")
            self.clear()
        self.update()

    def update_pix_for_grabcut(self):
        """更新显示以可视化 GrabCut 的内部状态 (PR_FG, PR_BGD)。"""
        if self.base_img is None or self.gc_mask is None or not self.gc_initialized:
            # If not in grabcut or something is wrong, fall back to standard update
            self.update_pix()
            return

        overlay = self.base_img.copy()
        h, w = overlay.shape[:2]
        if self.gc_mask.shape[:2] != (h, w):
            logging.error(f"GrabCut mask shape {self.gc_mask.shape[:2]} doesn't match image shape {(h, w)} during visualization.")
            self.update_pix() # Fallback
            return
            
        # Create visualization: Blue for PR_FG, Red for PR_BGD
        # Keep definite FG (1) as original, definite BG (0) as dimmed
        # Note: base_img is RGBA
        viz = overlay.copy()
        # Dim definite background
        viz[self.gc_mask == cv2.GC_BGD] = [120, 120, 120, 128] 
        # Overlay probable background with semi-transparent red
        pr_bgd_mask = (self.gc_mask == cv2.GC_PR_BGD)
        if np.any(pr_bgd_mask):
            red_overlay = np.zeros_like(overlay)
            red_overlay[:, :] = [255, 0, 0, 100] # Semi-transparent red
            viz[pr_bgd_mask] = cv2.addWeighted(overlay[pr_bgd_mask], 0.7, red_overlay[pr_bgd_mask], 0.3, 0)
        # Overlay probable foreground with semi-transparent blue
        pr_fgd_mask = (self.gc_mask == cv2.GC_PR_FGD)
        if np.any(pr_fgd_mask):
            blue_overlay = np.zeros_like(overlay)
            blue_overlay[:, :] = [0, 0, 255, 100] # Semi-transparent blue
            viz[pr_fgd_mask] = cv2.addWeighted(overlay[pr_fgd_mask], 0.7, blue_overlay[pr_fgd_mask], 0.3, 0)

        try:
            qimg = ImageQt.ImageQt(Image.fromarray(viz))
            pixmap = QPixmap.fromImage(qimg)
            # 统一处理方式，不再区分是否有缩放/平移
            transform = QtGui.QTransform()
            transform.scale(self.zoom_factor, self.zoom_factor)
            pixmap = pixmap.transformed(transform, Qt.TransformationMode.SmoothTransformation)
            result_pixmap = QPixmap(self.size())
            result_pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(result_pixmap)
            x = (self.width() - pixmap.width()) / 2 + self.pan_offset.x()
            y = (self.height() - pixmap.height()) / 2 + self.pan_offset.y()
            painter.drawPixmap(int(x), int(y), pixmap)
            painter.end()
            self._pixmap = result_pixmap
        except Exception as e:
            logging.exception(f"Error updating GrabCut visualization pixmap: {e}")
            self.clear()
        self.update()

    def reset_view(self):
        """重置缩放和平移状态到默认值"""
        self.zoom_factor = 1.0
        self.pan_offset = QtCore.QPoint(0, 0)
        self.update_pix()
        self.update()

    def set_mask(self, mask, reset_grabcut=True):
        """直接设置新的蒙版，并更新显示和历史记录。

        Args:
            mask (np.ndarray): 新的蒙版数据。
            reset_grabcut (bool, optional): 是否在设置新蒙版时重置 GrabCut 状态. Defaults to True.
        """
        # Always reset GrabCut state when mask is set externally or reset/cleared
        self.gc_initialized = False
        self.gc_mask = None
        # Keep models? No, likely inconsistent. Clear them.
        self.gc_bgd_model = None
        self.gc_fgd_model = None
        if self.mode == 'grabcut_refine':
            self.set_mode('draw') # Switch back to draw mode

        # Ensure incoming mask matches base image dimensions
        if self.base_img is not None and self.base_img.shape[:2] != mask.shape[:2]:
             logging.warning(f"set_mask received mask with shape {mask.shape[:2]}, resizing to match image {self.base_img.shape[:2]}.")
             self.mask = cv2.resize(mask, (self.base_img.shape[1], self.base_img.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            self.mask = mask.copy()
             
        self.push_history() # 添加到历史
        self.update_pix() # 更新显示
        self.mask_edited.emit(self.mask) # 发出信号

    def push_history(self):
        """将当前蒙版状态添加到历史记录栈。

        如果当前历史指针不在栈顶（即执行过撤销），则清除后续历史。
        限制历史记录的最大长度。
        """
        # 确保mask有效
        if self.mask is None:
             return
             
        # 清除redo历史
        if self.history_idx < len(self.history) - 1:
            self.history = self.history[:self.history_idx + 1]
        
        # 添加当前状态
        self.history.append(self.mask.copy())
        self.history_idx += 1

        # 限制历史记录大小
        MAX_HISTORY_SIZE = 32
        if len(self.history) > MAX_HISTORY_SIZE:
            self.history = self.history[-MAX_HISTORY_SIZE:]
            self.history_idx = len(self.history) - 1 # 更新指针

    def undo(self):
        """撤销操作：恢复到上一个历史状态，并重置 GrabCut 状态。"""
        if self.history_idx > 0:
            self.history_idx -= 1
            self.mask = self.history[self.history_idx].copy()
            
            # Reset GrabCut state on undo
            self.gc_initialized = False
            self.gc_mask = None
            if self.mode == 'grabcut_refine':
                self.set_mode('draw')
            else:
                 self.update_pix() # Standard update if not exiting refine mode
                 
            self.mask_edited.emit(self.mask)
            return True
        return False

    def redo(self):
        """重做操作：恢复到下一个历史状态，并重置 GrabCut 状态。"""
        if self.history_idx < len(self.history) - 1:
            self.history_idx += 1
            self.mask = self.history[self.history_idx].copy()

            # Reset GrabCut state on redo
            self.gc_initialized = False
            self.gc_mask = None
            if self.mode == 'grabcut_refine':
                self.set_mode('draw')
            else:
                self.update_pix()

            self.mask_edited.emit(self.mask)
            return True
        return False

    def edge_detect_canny(self, thresh1=50, thresh2=150, 
                          dilate_k=3, dilate_iter=1, 
                          close_k=3, close_iter=3):
        """
        使用 Canny 边缘检测和形态学操作生成蒙版 (参数化)。
        
        Args:
            thresh1 (int): Canny 边缘检测的低阈值。
            thresh2 (int): Canny 边缘检测的高阈值。
            dilate_k (int): 膨胀操作的核大小。
            dilate_iter (int): 膨胀操作的迭代次数。
            close_k (int): 闭运算操作的核大小。
            close_iter (int): 闭运算操作的迭代次数。
        """
        if self.base_img is None:
            logging.warning("Cannot run edge detection, base image is None.")
            return
        
        logging.info(f"Running edge_detect_canny: t1={thresh1}, t2={thresh2}, dk={dilate_k}, di={dilate_iter}, ck={close_k}, ci={close_iter}")
            
        try:
            if self.base_img.shape[2] == 4:
                gray = cv2.cvtColor(self.base_img[:,:,:3], cv2.COLOR_RGB2GRAY)
            else:
                gray = cv2.cvtColor(self.base_img, cv2.COLOR_RGB2GRAY)
            
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, thresh1, thresh2)
            
            if dilate_iter > 0 and dilate_k > 0:
                dilate_kernel = np.ones((int(dilate_k), int(dilate_k)), np.uint8)
                edges = cv2.dilate(edges, dilate_kernel, iterations=int(dilate_iter))
            
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, contours, -1, 255, -1)
            
            if close_iter > 0 and close_k > 0:
                close_kernel = np.ones((int(close_k), int(close_k)), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=int(close_iter))
            
            self.set_mask(mask)
        except Exception as e:
            logging.exception(f"Error during Canny edge detection: {e}")

    def auto_fix_morph(self, open_k=3, open_iter=1, close_k=3, close_iter=2):
        """自动修复蒙版 - 参数化的形态学操作。"""
        if self.mask is None:
            logging.warning("Cannot run auto fix, mask is None.")
            return
        
        logging.info(f"Running auto_fix_morph: ok={open_k}, oi={open_iter}, ck={close_k}, ci={close_iter}")
        
        try:
            current_mask = self.mask.copy()
            
            # Open operation (remove noise)
            if open_iter > 0 and open_k > 0:
                open_kernel = np.ones((int(open_k), int(open_k)), np.uint8)
                current_mask = cv2.morphologyEx(current_mask, cv2.MORPH_OPEN, open_kernel, iterations=int(open_iter))
            
            # Close operation (fill holes)
            if close_iter > 0 and close_k > 0:
                close_kernel = np.ones((int(close_k), int(close_k)), np.uint8)
                current_mask = cv2.morphologyEx(current_mask, cv2.MORPH_CLOSE, close_kernel, iterations=int(close_iter))
            
            self.set_mask(current_mask)
        except Exception as e:
            logging.exception(f"Error during auto fix morphology: {e}")

    def run_adaptive_threshold(self, adaptive_method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, block_size=11, C=2):
        """
        使用自适应阈值生成蒙版。
        
        Args:
            adaptive_method (int): 自适应阈值方法。
            block_size (int): 块大小。
            C (int): 常数。
        """
        if self.base_img is None:
            logging.warning("Cannot run adaptive threshold, base image is None.")
            return
            
        logging.info(f"Running adaptive threshold: method={adaptive_method}, block={block_size}, C={C}")
        
        try:
            if self.base_img.shape[2] == 4:
                gray = cv2.cvtColor(self.base_img[:,:,:3], cv2.COLOR_RGB2GRAY)
            else:
                gray = cv2.cvtColor(self.base_img, cv2.COLOR_RGB2GRAY)
                
            # Apply adaptive threshold
            thresh_mask = cv2.adaptiveThreshold(gray, 255,
                                                adaptive_method,
                                                cv2.THRESH_BINARY, 
                                                block_size, 
                                                C)
                                                
            # Adaptive threshold typically finds dark objects on light background (object=0, bg=255)
            # We need the opposite (object=255, bg=0)
            final_mask = cv2.bitwise_not(thresh_mask)
            
            self.set_mask(final_mask)
        except Exception as e:
            logging.exception(f"Error during adaptive threshold: {e}")

    def paintEvent(self, event):
        """自定义绘制，确保内容居中显示，并绘制 GrabCut 矩形框。"""
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        
        # Draw the base image/mask pixmap first
        if self._pixmap is not None:
            widget_w, widget_h = self.width(), self.height()
            pix_w, pix_h = self._pixmap.width(), self._pixmap.height()
            x = (widget_w - pix_w) // 2
            y = (widget_h - pix_h) // 2
            painter.drawPixmap(x, y, self._pixmap)
            
        # Draw the GrabCut rectangle if currently drawing
        if self.drawing_rect and self.grabcut_rect:
            painter.setPen(QtGui.QPen(QColor(255, 0, 0), 1, Qt.PenStyle.DashLine)) # Red dashed line
            painter.drawRect(self.grabcut_rect)
            
        painter.end()

    def keyPressEvent(self, event):
        """处理键盘事件，实现快捷键功能"""
        # 检查是否按下了Ctrl键
        ctrl_pressed = event.modifiers() & Qt.KeyboardModifier.ControlModifier
        
        # 处理Ctrl+Z (撤销) 和 Ctrl+Y (重做)
        if ctrl_pressed:
            if event.key() == QtCore.Qt.Key.Key_Z:
                self.undo()
                event.accept()
                return
            elif event.key() == QtCore.Qt.Key.Key_Y:
                self.redo()
                event.accept()
                return
        
        # 处理其他快捷键
        if event.key() in self.shortcuts:
            self.shortcuts[event.key()]()
            event.accept()
            return
            
        # 如果没有处理事件，则传递给父类
        super().keyPressEvent(event)
    
    def clear_mask(self):
        """清除整个蒙版（设置为0），并重置 GrabCut 状态。"""
        if self.mask is not None:
            # 保存当前状态到历史
            old_mask = self.mask.copy()
            
            # 清除蒙版
            self.mask.fill(0)
            
            # 更新历史和显示
            self.push_history()
            # Reset GrabCut state when clearing
            self.gc_initialized = False
            self.gc_mask = None
            if self.mode == 'grabcut_refine':
                self.set_mode('draw')
            else:
                self.update_pix()
                
            self.mask_edited.emit(self.mask)
    
    def increase_brush_size(self):
        """增加画笔大小"""
        self.brush_size = min(100, self.brush_size + 5)  # 最大100
    
    def decrease_brush_size(self):
        """减小画笔大小"""
        self.brush_size = max(1, self.brush_size - 5)  # 最小1
    
    def reset_zoom(self):
        """重置缩放到原始大小"""
        self.zoom_factor = 1.0
        self.pan_offset = QtCore.QPoint(0, 0)
        self.update_pix()

    def clear(self):
        """清空显示内容"""
        self._pixmap = None
        self.update()

    def run_watershed_segmentation(self):
        """ 基于用户标记执行 Watershed 分割。"""
        logging.info("Running Watershed segmentation...")
        if self.watershed_markers is None or self.base_img is None:
            logging.error("Watershed markers or base image not available for segmentation!")
            if self.watershed_markers is None:
                logging.error("watershed_markers is None")
            if self.base_img is None:
                logging.error("base_img is None")
            return
            
        # Check if markers have been added (at least one FG and one BG?)
        unique_markers = np.unique(self.watershed_markers)
        logging.info(f"Unique marker values found: {unique_markers}")
        if 1 not in unique_markers or 2 not in unique_markers:
            logging.warning("Missing required markers: need both FG(1) and BG(2) markers")
            QtWidgets.QMessageBox.warning(self.parent_dialog, "标记不足", 
                                           "请至少标记一些前景区域 (1) 和背景区域 (2)。")
            return

        logging.info("Starting Watershed algorithm calculation...")
        
        try:
            # Ensure image is BGR for watershed
            if self.base_img.shape[2] == 4:
                img_bgr = cv2.cvtColor(self.base_img[:, :, :3], cv2.COLOR_RGBA2BGR)
            else:
                img_bgr = cv2.cvtColor(self.base_img[:, :, :3].copy(), cv2.COLOR_RGB2BGR)

            # Create a copy of markers for the algorithm
            markers_copy = self.watershed_markers.copy()
            logging.debug(f"Watershed markers shape: {markers_copy.shape}, img_bgr shape: {img_bgr.shape}")

            # Run watershed
            logging.info("Calling cv2.watershed()...")
            cv2.watershed(img_bgr, markers_copy)
            logging.info("cv2.watershed() completed")
            
            # Generate binary mask: -1 are boundaries, 1 is FG, others are BG
            # We want areas marked originally as FG (1) that didn't become boundaries (-1)
            output_mask = np.zeros_like(self.mask, dtype=np.uint8)
            output_mask[markers_copy == 1] = 255 # Mark watershed FG regions as 255
            logging.debug(f"Generated output mask with {np.sum(output_mask > 0)} foreground pixels")
            # Optional: Treat boundaries as background (might remove thin foreground parts)
            # output_mask[markers_copy == -1] = 0 
            
            # Apply the generated mask
            logging.info("Setting final watershed segmentation mask")
            self.set_mask(output_mask)
            logging.info("Watershed segmentation finished.")

        except Exception as e:
            logging.exception(f"Error during Watershed execution: {e}")
            QtWidgets.QMessageBox.critical(self.parent_dialog, "分割出错", f"执行分水岭算法时出错: {e}")
        finally:
            # Exit watershed mode regardless of success or failure
            logging.info("Exiting watershed mode")
            self.set_mode('draw')
            self.watershed_markers = None # Clear markers

    def update_pix_for_watershed(self):
        """更新显示以可视化 Watershed 标记。"""
        logging.debug("Entered update_pix_for_watershed")
        if self.base_img is None or self.watershed_markers is None:
            logging.debug(f"Fallback to standard: base_img is None? {self.base_img is None}, watershed_markers is None? {self.watershed_markers is None}")
            self.update_pix() # Fallback to standard if no markers
            return

        logging.debug(f"Base img shape: {self.base_img.shape if self.base_img is not None else 'None'}")
        logging.debug(f"Watershed markers shape: {self.watershed_markers.shape if self.watershed_markers is not None else 'None'}")
        
        viz = self.base_img.copy()
        h, w = viz.shape[:2]
        if self.watershed_markers.shape[:2] != (h, w):
             logging.error(f"Watershed marker shape mismatch: markers={self.watershed_markers.shape[:2]}, image={viz.shape[:2]}")
             self.update_pix()
             return

        # Overlay markers: Blue for FG (1), Red for BG (2)
        fg_mask = (self.watershed_markers == 1)
        bg_mask = (self.watershed_markers == 2)
        
        logging.debug(f"FG markers count: {np.sum(fg_mask)}")
        logging.debug(f"BG markers count: {np.sum(bg_mask)}")

        # 使用更鲜明的颜色进行调试 - 透明度降低以更明显地看到标记
        if np.any(fg_mask):
            blue_overlay = np.zeros_like(viz)
            blue_overlay[:, :] = [0, 0, 255, 200] # Brighter blue
            viz[fg_mask] = cv2.addWeighted(viz[fg_mask], 0.3, blue_overlay[fg_mask], 0.7, 0)
            # 简化调试：使用纯色而不是半透明
            # viz[fg_mask] = [0, 0, 255, 255]  # 纯蓝色

        if np.any(bg_mask):
            red_overlay = np.zeros_like(viz)
            red_overlay[:, :] = [255, 0, 0, 200] # Brighter red
            viz[bg_mask] = cv2.addWeighted(viz[bg_mask], 0.3, red_overlay[bg_mask], 0.7, 0)
            # 简化调试：使用纯色而不是半透明
            # viz[bg_mask] = [255, 0, 0, 255]  # 纯红色

        # Apply standard pixmap update logic (zoom, pan, set pixmap)
        try:
            logging.debug("Creating QImage from visualization")
            qimg = ImageQt.ImageQt(Image.fromarray(viz))
            pixmap = QPixmap.fromImage(qimg)
            # 统一处理方式，不再区分是否有缩放/平移
            transform = QtGui.QTransform()
            transform.scale(self.zoom_factor, self.zoom_factor)
            pixmap = pixmap.transformed(transform, Qt.TransformationMode.SmoothTransformation)
            result_pixmap = QPixmap(self.size())
            result_pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(result_pixmap)
            x = (self.width() - pixmap.width()) / 2 + self.pan_offset.x()
            y = (self.height() - pixmap.height()) / 2 + self.pan_offset.y()
            painter.drawPixmap(int(x), int(y), pixmap)
            painter.end()
            self._pixmap = result_pixmap
        except Exception as e:
            logging.exception(f"Error updating Watershed visualization pixmap: {e}")
            self.clear()
        self.update()