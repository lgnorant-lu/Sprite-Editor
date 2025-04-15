"""
---------------------------------------------------------------
File name:                  dialogs.py
Author:                     Ignorant-lu
Date created:               2025/04/15
Description:                Sprite Mask Editor工具的对话框组件
----------------------------------------------------------------

Changed history:            
                            2025/04/15: 从sprite_mask_editor.py拆分为独立模块;
----
"""
import numpy as np
import cv2
from PyQt6 import QtWidgets, QtGui, QtCore
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QIcon, QPixmap, QShortcut, QKeySequence, QAction
import logging

from PIL import Image, ImageQt

from .roi import FrameROI
from .widgets import MaskEditWidget
from .constants import DEFAULT_BRUSH_SIZE

# --- Parameter Dialogs Start ---
class MorphologyParamsDialog(QtWidgets.QDialog):
    """用于设置形态学操作参数的对话框。"""
    def __init__(self, parent=None, 
                 default_open_k=3, default_open_iter=1, 
                 default_close_k=3, default_close_iter=2):
        super().__init__(parent)
        self.setWindowTitle("形态学参数")
        layout = QtWidgets.QFormLayout(self)

        # Open Operation
        open_group = QtWidgets.QGroupBox("开运算 (去噪点)")
        open_group.setObjectName("openGroup")
        open_layout = QtWidgets.QFormLayout(open_group)
        self.open_kernel_spin = QtWidgets.QSpinBox()
        self.open_kernel_spin.setRange(1, 15) # Kernel size (odd numbers often preferred)
        self.open_kernel_spin.setSingleStep(2)
        self.open_kernel_spin.setValue(default_open_k)
        self.open_kernel_spin.setToolTip("开运算核大小：影响去噪点操作的范围，越大去除的噪点越大。")
        self.open_iter_spin = QtWidgets.QSpinBox()
        self.open_iter_spin.setRange(0, 10)
        self.open_iter_spin.setValue(default_open_iter)
        self.open_iter_spin.setToolTip("开运算迭代次数：重复去噪点操作的次数，越多效果越强。")
        open_layout.addRow("核大小:", self.open_kernel_spin)
        open_layout.addRow("迭代次数:", self.open_iter_spin)
        layout.addRow(open_group)

        # Close Operation
        close_group = QtWidgets.QGroupBox("闭运算 (填洞)")
        close_group.setObjectName("closeGroup")
        close_layout = QtWidgets.QFormLayout(close_group)
        self.close_kernel_spin = QtWidgets.QSpinBox()
        self.close_kernel_spin.setRange(1, 15)
        self.close_kernel_spin.setSingleStep(2)
        self.close_kernel_spin.setValue(default_close_k)
        self.close_kernel_spin.setToolTip("闭运算核大小：影响填补孔洞操作的范围，越大填补的孔洞越大。")
        self.close_iter_spin = QtWidgets.QSpinBox()
        self.close_iter_spin.setRange(0, 10)
        self.close_iter_spin.setValue(default_close_iter)
        self.close_iter_spin.setToolTip("闭运算迭代次数：重复填补孔洞操作的次数，越多效果越强。")
        close_layout.addRow("核大小:", self.close_kernel_spin)
        close_layout.addRow("迭代次数:", self.close_iter_spin)
        layout.addRow(close_group)

        # Dialog buttons
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok |
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def get_params(self):
        """获取用户设置的参数。"""
        return {
            'open_k': self.open_kernel_spin.value(),
            'open_iter': self.open_iter_spin.value(),
            'close_k': self.close_kernel_spin.value(),
            'close_iter': self.close_iter_spin.value()
        }

class CannyParamsDialog(MorphologyParamsDialog):
    """用于设置Canny边缘检测和后续形态学操作参数的对话框。"""
    def __init__(self, parent=None, default_thresh1=50, default_thresh2=150,
                 default_dilate_k=3, default_dilate_iter=1,
                 default_close_k=3, default_close_iter=3): # Inherits morphology params
        # Initialize morphology part first (using different defaults for Canny post-processing)
        super().__init__(parent, 
                         default_open_k=default_dilate_k, # Reuse open_k for dilate_k UI
                         default_open_iter=default_dilate_iter, # Reuse open_iter for dilate_iter UI
                         default_close_k=default_close_k, 
                         default_close_iter=default_close_iter)
        self.setWindowTitle("Canny 边缘检测参数")

        # Rename morphology groups for clarity in Canny context - Use objectName
        open_group_widget = self.findChild(QtWidgets.QGroupBox, "openGroup")
        if open_group_widget:
            open_group_widget.setTitle("边缘膨胀 (连接断线)")
            open_group_widget.setToolTip("对检测到的边缘进行膨胀，尝试连接断开的线条。")
        close_group_widget = self.findChild(QtWidgets.QGroupBox, "closeGroup")
        if close_group_widget:
            close_group_widget.setTitle("轮廓闭合 (填充内部)")
            close_group_widget.setToolTip("对膨胀后的轮廓进行闭运算，尝试填充内部的小孔洞，形成完整区域。")

        # Canny Thresholds Group
        canny_group = QtWidgets.QGroupBox("Canny 阈值")
        canny_group.setToolTip("Canny 边缘检测算法的核心阈值。")
        canny_layout = QtWidgets.QFormLayout(canny_group)
        self.thresh1_spin = QtWidgets.QSpinBox()
        self.thresh1_spin.setRange(0, 500)
        self.thresh1_spin.setValue(default_thresh1)
        self.thresh1_spin.setToolTip("低阈值：低于此值的边会被抑制。")
        self.thresh2_spin = QtWidgets.QSpinBox()
        self.thresh2_spin.setRange(0, 1000)
        self.thresh2_spin.setValue(default_thresh2)
        self.thresh2_spin.setToolTip("高阈值：高于此值的边被视为强边缘。介于两者之间的边，只有连接到强边缘才保留。")
        canny_layout.addRow("低阈值 (Threshold1):", self.thresh1_spin)
        canny_layout.addRow("高阈值 (Threshold2):", self.thresh2_spin)

        # Insert Canny group before morphology groups in the main layout
        main_layout = self.layout()
        main_layout.insertRow(0, canny_group)

    def get_params(self):
        """获取Canny和形态学参数。"""
        morph_params = super().get_params() # Get morphology params
        return {
            'thresh1': self.thresh1_spin.value(),
            'thresh2': self.thresh2_spin.value(),
            # Map back from reused morphology spins
            'dilate_k': morph_params['open_k'], 
            'dilate_iter': morph_params['open_iter'],
            'close_k': morph_params['close_k'],
            'close_iter': morph_params['close_iter']
        }
# --- Parameter Dialogs End ---

# --- Adaptive Threshold Dialog Start ---
class AdaptiveThresholdParamsDialog(QtWidgets.QDialog):
    """用于设置自适应阈值参数的对话框。"""
    def __init__(self, parent=None, default_method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                 default_block_size=11, default_c=2):
        super().__init__(parent)
        self.setWindowTitle("自适应阈值参数")
        layout = QtWidgets.QFormLayout(self)

        # Adaptive Method
        self.method_combo = QtWidgets.QComboBox()
        self.method_combo.addItem("高斯加权均值 (Gaussian C)", cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
        self.method_combo.addItem("邻域均值 (Mean C)", cv2.ADAPTIVE_THRESH_MEAN_C)
        current_index = self.method_combo.findData(default_method)
        if current_index >= 0:
            self.method_combo.setCurrentIndex(current_index)
        layout.addRow("自适应方法:", self.method_combo)

        # Block Size
        self.block_size_spin = QtWidgets.QSpinBox()
        self.block_size_spin.setRange(3, 255) # Block size must be odd
        self.block_size_spin.setSingleStep(2)
        self.block_size_spin.setValue(default_block_size)
        self.block_size_spin.setToolTip("计算阈值的像素邻域大小，必须是奇数。")
        layout.addRow("块大小 (奇数):", self.block_size_spin)

        # Constant C
        self.c_spin = QtWidgets.QSpinBox()
        self.c_spin.setRange(-50, 50) # C can be positive or negative
        self.c_spin.setValue(default_c)
        self.c_spin.setToolTip("从计算出的均值或加权和中减去的常数。")
        layout.addRow("常数 C:", self.c_spin)

        # Dialog buttons
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok |
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def get_params(self):
        """获取用户设置的参数。"""
        block_size = self.block_size_spin.value()
        # Ensure block size is odd
        if block_size % 2 == 0:
            block_size += 1 
            logging.warning(f"Block size was even, adjusted to {block_size}")
            
        return {
            'adaptive_method': self.method_combo.currentData(),
            'block_size': block_size,
            'C': self.c_spin.value()
        }
# --- Adaptive Threshold Dialog End ---

class MaskEditDialog(QtWidgets.QDialog):
    """蒙版编辑对话框，用于编辑ROI的蒙版。

    提供画笔/橡皮工具、撤销/重做功能和自动修复功能。
    """

    def __init__(self, parent, img, mask):
        """初始化MaskEditDialog。

        Args:
            parent (QWidget): 父控件
            img (np.ndarray): 原始图像数据 (RGBA)
            mask (np.ndarray): 初始蒙版数据 (灰度图, H, W)
        """
        super().__init__(parent)
        self.setWindowTitle("蒙版编辑")
        self.setWindowFlag(Qt.WindowType.WindowContextHelpButtonHint, False)
        self.setMinimumSize(600, 500)

        # 获取当前掩码和原始图像
        self.original_img = img
        self.original_mask = mask

        # 确保掩码尺寸与图像一致
        if self.original_mask.shape[:2] != self.original_img.shape[:2]:
            self.original_mask = cv2.resize(
                self.original_mask, 
                (self.original_img.shape[1], self.original_img.shape[0]), 
                interpolation=cv2.INTER_NEAREST
            )

        # 创建UI
        self.init_ui()
        
        # 设置键盘快捷键
        self.setup_shortcuts()

    def init_ui(self):
        """初始化界面布局和控件。"""
        layout = QtWidgets.QVBoxLayout()
        
        # 创建工具栏
        self.toolbar = QtWidgets.QToolBar()
        self.toolbar.setIconSize(QtCore.QSize(24, 24))
        
        # 工具按钮
        self.btn_draw = QtWidgets.QToolButton()
        self.btn_draw.setText("绘制")
        self.btn_draw.setIcon(QIcon(":/icons/draw.png"))
        self.btn_draw.setToolTip("绘制蒙版 [B]")
        self.btn_draw.setCheckable(True)
        self.btn_draw.setChecked(True)
        self.toolbar.addWidget(self.btn_draw)
        
        self.btn_erase = QtWidgets.QToolButton()
        self.btn_erase.setText("擦除")
        self.btn_erase.setIcon(QIcon(":/icons/erase.png"))
        self.btn_erase.setToolTip("擦除蒙版 [E]")
        self.btn_erase.setCheckable(True)
        self.toolbar.addWidget(self.btn_erase)
        
        # --- Separator 1 ---        
        sep1_action = self.toolbar.addSeparator()
 
        # --- GrabCut Tools Frame (Initially Hidden) ---
        self.gc_tools_frame = QtWidgets.QFrame()
        gc_layout = QtWidgets.QHBoxLayout(self.gc_tools_frame)
        gc_layout.setContentsMargins(5, 0, 5, 0)
        gc_layout.setSpacing(5)

        self.btn_gc_fg = QtWidgets.QToolButton()
        self.btn_gc_fg.setText("前景标记")
        self.btn_gc_fg.setIcon(QIcon(":/icons/fg_marker.png"))
        self.btn_gc_fg.setToolTip("标记确定为前景的区域")
        self.btn_gc_fg.setCheckable(True)
        gc_layout.addWidget(self.btn_gc_fg)

        self.btn_gc_bg = QtWidgets.QToolButton()
        self.btn_gc_bg.setText("背景标记")
        self.btn_gc_bg.setIcon(QIcon(":/icons/bg_marker.png"))
        self.btn_gc_bg.setToolTip("标记确定为背景的区域")
        self.btn_gc_bg.setCheckable(True)
        gc_layout.addWidget(self.btn_gc_bg)

        self.gc_marker_group = QtWidgets.QButtonGroup(self)
        self.gc_marker_group.addButton(self.btn_gc_fg)
        self.gc_marker_group.addButton(self.btn_gc_bg)
        self.gc_marker_group.setExclusive(True)

        self.btn_gc_finish = QtWidgets.QToolButton()
        self.btn_gc_finish.setText("完成分割")
        self.btn_gc_finish.setIcon(QIcon(":/icons/check.png"))
        self.btn_gc_finish.setToolTip("完成 GrabCut 分割并应用蒙版")
        gc_layout.addWidget(self.btn_gc_finish)

        self.gc_tools_frame.setVisible(False)
        self.toolbar.insertWidget(sep1_action, self.gc_tools_frame)
        # --- End GrabCut Tools Frame ---

        # --- Watershed Tools Frame (Initially Hidden) ---
        self.ws_tools_frame = QtWidgets.QFrame()
        # 明确设置最小尺寸，确保即使内容较小也能显示
        self.ws_tools_frame.setMinimumWidth(200)
        self.ws_tools_frame.setMinimumHeight(40)
        # 添加明显的边框和背景色，帮助调试
        self.ws_tools_frame.setStyleSheet("QFrame { border: 1px solid #3498db; background-color: #eef6ff; border-radius: 3px; }")
        self.ws_layout = QtWidgets.QHBoxLayout(self.ws_tools_frame)
        self.ws_layout.setContentsMargins(5, 0, 5, 0)
        self.ws_layout.setSpacing(5)

        self.btn_ws_fg = QtWidgets.QToolButton()
        self.btn_ws_fg.setText("前景标记")
        self.btn_ws_fg.setIcon(QIcon(":/icons/ws_fg.png"))
        self.btn_ws_fg.setToolTip("标记确定前景")
        self.btn_ws_fg.setCheckable(True)
        self.btn_ws_fg.setVisible(False)  # 初始隐藏
        self.ws_layout.addWidget(self.btn_ws_fg)

        self.btn_ws_bg = QtWidgets.QToolButton()
        self.btn_ws_bg.setText("背景标记")
        self.btn_ws_bg.setIcon(QIcon(":/icons/ws_bg.png"))
        self.btn_ws_bg.setToolTip("标记确定背景")
        self.btn_ws_bg.setCheckable(True)
        self.btn_ws_bg.setVisible(False)  # 初始隐藏
        self.ws_layout.addWidget(self.btn_ws_bg)

        self.ws_marker_group = QtWidgets.QButtonGroup(self)
        self.ws_marker_group.addButton(self.btn_ws_fg)
        self.ws_marker_group.addButton(self.btn_ws_bg)
        self.ws_marker_group.setExclusive(True)

        self.btn_ws_run = QtWidgets.QToolButton()
        self.btn_ws_run.setText("执行分割")
        self.btn_ws_run.setIcon(QIcon(":/icons/run.png"))
        self.btn_ws_run.setToolTip("基于标记执行分水岭分割")
        self.btn_ws_run.setVisible(False)  # 初始隐藏
        self.ws_layout.addWidget(self.btn_ws_run)

        self.toolbar.insertWidget(sep1_action, self.ws_tools_frame)
        # --- End Watershed Tools Frame ---
        
        # Separator before standard buttons - REMOVED (sep1_action already added)
        # toolbar.addSeparator()
        
        # Undo/Redo buttons (Added AFTER sep1_action implicitly)
        self.btn_undo = QtWidgets.QToolButton()
        self.btn_undo.setText("撤销")
        self.btn_undo.setIcon(QIcon(":/icons/undo.png"))
        self.btn_undo.setToolTip("撤销上一步 [Ctrl+Z]")
        self.toolbar.addWidget(self.btn_undo)
        
        self.btn_redo = QtWidgets.QToolButton()
        self.btn_redo.setText("重做")
        self.btn_redo.setIcon(QIcon(":/icons/redo.png"))
        self.btn_redo.setToolTip("重做下一步 [Ctrl+Y]")
        self.toolbar.addWidget(self.btn_redo)
        
        self.toolbar.addSeparator() # Separator 2
        
        # 重置蒙版按钮
        self.btn_reset = QtWidgets.QToolButton()
        self.btn_reset.setText("重置")
        self.btn_reset.setIcon(QIcon(":/icons/trash.png"))
        self.btn_reset.setToolTip("重置蒙版 [R]")
        self.toolbar.addWidget(self.btn_reset)
        
        # 自动修复按钮 - 连接到新的参数化运行函数
        self.btn_auto_fix = QtWidgets.QToolButton()
        self.btn_auto_fix.setText("自动修复")
        self.btn_auto_fix.setIcon(QIcon(":/icons/auto.png"))
        self.btn_auto_fix.setToolTip("自动修复蒙版（参数可调）[A]")
        self.btn_auto_fix.clicked.connect(self.run_auto_fix)
        self.toolbar.addWidget(self.btn_auto_fix)

        # 自动蒙版按钮 (替换边缘检测)
        self.btn_auto_mask = QtWidgets.QToolButton()
        self.btn_auto_mask.setText("自动蒙版")
        self.btn_auto_mask.setIcon(QIcon(":/icons/magic.png"))
        self.btn_auto_mask.setToolTip("使用算法自动生成蒙版 [D]")
        self.btn_auto_mask.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        
        auto_mask_menu = QtWidgets.QMenu(self)
        
        canny_action = QAction("Canny 边缘检测", self)
        canny_action.triggered.connect(self.run_canny)
        auto_mask_menu.addAction(canny_action)
        
        # Add action for Adaptive Threshold
        adaptive_action = QAction("自适应阈值", self)
        adaptive_action.triggered.connect(self.run_adaptive_threshold)
        auto_mask_menu.addAction(adaptive_action)
        
        # Add action for GrabCut
        grabcut_action = QAction("GrabCut 分割", self)
        grabcut_action.setToolTip("通过绘制矩形框选前景进行分割")
        grabcut_action.triggered.connect(self.run_grabcut)
        auto_mask_menu.addAction(grabcut_action)

        # Add action for Watershed
        watershed_action = QAction("分水岭 分割", self)
        watershed_action.setToolTip("通过标记前景/背景区域进行分割")
        watershed_action.triggered.connect(self.run_watershed)
        auto_mask_menu.addAction(watershed_action)
        
        self.btn_auto_mask.setMenu(auto_mask_menu)
        self.btn_auto_mask.clicked.connect(self.run_canny)
        self.toolbar.addWidget(self.btn_auto_mask)

        # 添加重置视图按钮
        self.btn_reset_view = QtWidgets.QToolButton()
        self.btn_reset_view.setText("重置视图")
        self.btn_reset_view.setIcon(QIcon(":/icons/fit.png"))
        self.btn_reset_view.setToolTip("重置缩放和平移 [Home]")
        self.toolbar.addWidget(self.btn_reset_view)
        
        self.toolbar.addSeparator()
        
        # 笔刷大小控制
        self.toolbar.addWidget(QtWidgets.QLabel("笔刷大小:"))
        
        self.brush_size = QtWidgets.QSpinBox()
        self.brush_size.setMinimum(1)
        self.brush_size.setMaximum(100)
        self.brush_size.setValue(DEFAULT_BRUSH_SIZE)
        self.brush_size.setToolTip("调整笔刷大小 [[ 减小 | ]] 增大")
        self.toolbar.addWidget(self.brush_size)

        # 添加快捷键帮助按钮
        self.toolbar.addSeparator()
        self.btn_help = QtWidgets.QToolButton()
        self.btn_help.setText("快捷键")
        self.btn_help.setIcon(QIcon(":/icons/help.png"))
        self.btn_help.setToolTip("显示快捷键帮助")
        self.toolbar.addWidget(self.btn_help)
        
        layout.addWidget(self.toolbar)
        
        # 主编辑区域
        self.edit_widget = MaskEditWidget(self.original_img, self.original_mask, self)
        layout.addWidget(self.edit_widget)
        
        # 底部按钮
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok |
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        layout.addWidget(buttons)
        
        self.setLayout(layout)
        
        # 连接信号
        self.btn_draw.clicked.connect(lambda: self.edit_widget.set_mode('draw'))
        self.btn_erase.clicked.connect(lambda: self.edit_widget.set_mode('erase'))
        self.btn_undo.clicked.connect(self.edit_widget.undo)
        self.btn_redo.clicked.connect(self.edit_widget.redo)
        self.btn_reset.clicked.connect(self.reset_mask)
        self.btn_reset_view.clicked.connect(self.edit_widget.reset_view)
        self.btn_help.clicked.connect(self.show_shortcuts_help)
        self.brush_size.valueChanged.connect(self.edit_widget.set_brush_size)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        # GrabCut marker buttons
        self.btn_gc_fg.clicked.connect(lambda: self.edit_widget.set_grabcut_marker_mode('fg'))
        self.btn_gc_bg.clicked.connect(lambda: self.edit_widget.set_grabcut_marker_mode('bg'))
        self.btn_gc_finish.clicked.connect(self.edit_widget.finish_grabcut)
        # Watershed buttons
        self.btn_ws_run.clicked.connect(self.edit_widget.run_watershed_segmentation)

        # Connect the signal AFTER adding buttons
        self.ws_marker_group.buttonClicked.connect(self.on_watershed_marker_selected)

    def setup_shortcuts(self):
        """设置键盘快捷键"""
        # 画笔模式
        shortcut_draw = QShortcut(QKeySequence("B"), self)
        shortcut_draw.activated.connect(lambda: (self.btn_draw.setChecked(True), self.edit_widget.set_mode('draw')))
        
        # 橡皮模式
        shortcut_erase = QShortcut(QKeySequence("E"), self)
        shortcut_erase.activated.connect(lambda: (self.btn_erase.setChecked(True), self.edit_widget.set_mode('erase')))
        
        # 撤销/重做
        shortcut_undo = QShortcut(QKeySequence("Ctrl+Z"), self)
        shortcut_undo.activated.connect(self.edit_widget.undo)
        
        shortcut_redo = QShortcut(QKeySequence("Ctrl+Y"), self)
        shortcut_redo.activated.connect(self.edit_widget.redo)
        
        # 调整笔刷大小
        shortcut_brush_smaller = QShortcut(QKeySequence("["), self)
        shortcut_brush_smaller.activated.connect(self.decrease_brush_size)
        
        shortcut_brush_larger = QShortcut(QKeySequence("]"), self)
        shortcut_brush_larger.activated.connect(self.increase_brush_size)
        
        # 重置蒙版
        shortcut_reset = QShortcut(QKeySequence("R"), self)
        shortcut_reset.activated.connect(self.reset_mask)
        
        # 自动修复 - 连接到新槽函数
        shortcut_auto_fix = QShortcut(QKeySequence("A"), self)
        shortcut_auto_fix.activated.connect(self.run_auto_fix)
        
        # 保存
        shortcut_save = QShortcut(QKeySequence("Ctrl+S"), self)
        shortcut_save.activated.connect(self.accept)

        # 自动蒙版 (原边缘检测 D 键) - 连接到 Canny (或者可以连接到上次使用的?)
        shortcut_auto_mask = QShortcut(QKeySequence("D"), self)
        shortcut_auto_mask.activated.connect(self.run_canny)

        # 重置视图
        shortcut_reset_view = QShortcut(QKeySequence("Home"), self)
        shortcut_reset_view.activated.connect(self.edit_widget.reset_view)

    def keyPressEvent(self, event):
        """处理键盘事件"""
        # 处理Escape键关闭对话框
        if event.key() == Qt.Key.Key_Escape:
            self.reject()
        else:
            super().keyPressEvent(event)

    def decrease_brush_size(self):
        """减小笔刷大小"""
        current_size = self.brush_size.value()
        if current_size > 1:
            self.brush_size.setValue(current_size - 1)

    def increase_brush_size(self):
        """增大笔刷大小"""
        current_size = self.brush_size.value()
        if current_size < 100:
            self.brush_size.setValue(current_size + 1)

    def get_mask(self):
        """获取编辑后的蒙版。

        Returns:
            np.ndarray: 编辑后的蒙版。
        """
        return self.edit_widget.mask

    def reset_mask(self):
        """重置蒙版为全黑（清除所有）。"""
        result = QtWidgets.QMessageBox.question(
            self,
            "确认重置",
            "确定要重置蒙版吗？此操作不可撤销！",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No
        )
        if result == QtWidgets.QMessageBox.StandardButton.Yes:
            # 创建一个全黑蒙版
            blank_mask = np.zeros_like(self.original_mask)
            self.edit_widget.set_mask(blank_mask)

    def run_auto_fix(self):
        """显示形态学参数对话框并执行自动修复。"""
        dialog = MorphologyParamsDialog(self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            params = dialog.get_params()
            logging.info(f"Running Auto Fix with params: {params}")
            self.edit_widget.auto_fix_morph(**params) # Pass params as keyword args

    def run_canny(self):
        """显示Canny参数对话框并执行边缘检测。"""
        dialog = CannyParamsDialog(self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            params = dialog.get_params()
            logging.info(f"Running Canny edge detection with params: {params}")
            self.edit_widget.edge_detect_canny(**params) # Pass params as keyword args

    def run_adaptive_threshold(self):
        """显示自适应阈值参数对话框并执行。"""
        dialog = AdaptiveThresholdParamsDialog(self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            params = dialog.get_params()
            logging.info(f"Running Adaptive Threshold with params: {params}")
            self.edit_widget.run_adaptive_threshold(**params)

    def run_grabcut(self):
        """启动 GrabCut 流程：提示用户并进入矩形框选模式。"""
        QtWidgets.QMessageBox.information(self, "GrabCut 分割", 
                                          "请在图像上拖动鼠标，绘制一个紧密包围主要前景对象的矩形框。")
        self.edit_widget.enter_grabcut_rect_mode()

    def run_watershed(self):
        """启动 Watershed 流程：提示用户并进入标记模式。"""
        logging.info("Starting Watershed segmentation process")

        # 创建3个直接添加到工具栏的按钮 (临时测试方案)
        test_fg_btn = QtWidgets.QToolButton()
        test_fg_btn.setText("测试前景")
        test_fg_btn.setIcon(QIcon(":/icons/ws_fg.png"))
        test_fg_btn.setToolTip("测试前景标记按钮")
        test_fg_btn.setCheckable(True)
        
        test_bg_btn = QtWidgets.QToolButton()
        test_bg_btn.setText("测试背景")
        test_bg_btn.setIcon(QIcon(":/icons/ws_bg.png"))
        test_bg_btn.setToolTip("测试背景标记按钮")
        test_bg_btn.setCheckable(True)
        
        test_run_btn = QtWidgets.QToolButton()
        test_run_btn.setText("测试分割")
        test_run_btn.setIcon(QIcon(":/icons/run.png"))
        test_run_btn.setToolTip("测试执行分割按钮")

        # 用来保存按钮，防止被垃圾回收
        self.test_buttons = [test_fg_btn, test_bg_btn, test_run_btn]
        
        # 直接插入到工具栏的开头
        sep_action = self.toolbar.actions()[0]  # 获取第一个分隔符
        self.toolbar.insertWidget(sep_action, test_run_btn)
        self.toolbar.insertWidget(sep_action, test_bg_btn)
        self.toolbar.insertWidget(sep_action, test_fg_btn)
        
        # 连接测试按钮的信号
        test_fg_btn.clicked.connect(lambda: self.edit_widget.set_watershed_marker_mode('fg'))
        test_bg_btn.clicked.connect(lambda: self.edit_widget.set_watershed_marker_mode('bg'))
        test_run_btn.clicked.connect(self.edit_widget.run_watershed_segmentation)
        
        # 设置按钮组以确保互斥
        test_group = QtWidgets.QButtonGroup(self)
        test_group.addButton(test_fg_btn)
        test_group.addButton(test_bg_btn)
        test_group.setExclusive(True)
        test_fg_btn.setChecked(True)
        
        logging.debug("测试按钮已添加到工具栏")

        QtWidgets.QMessageBox.information(self, "分水岭 分割", 
                                          "请使用前景(蓝)/背景(红)画笔标记图像区域，然后点击\"执行分割\"。")
        logging.debug("Entering watershed mode")
        self.edit_widget.enter_watershed_mode()
        logging.debug("Watershed mode entered")

    def show_shortcuts_help(self):
        """显示快捷键和基本交互帮助信息"""
        help_text = (
            "<b>蒙版编辑器 - 帮助</b><br><br>"
            
            "<b>核心交互:</b><br>"
            "- <b>绘制/擦除/标记:</b> 在编辑区域按住鼠标左键拖动。<br>"
            "- <b>平移视图:</b> 按住 Shift 键并拖动鼠标左键。<br>"
            "- <b>缩放视图:</b> 使用鼠标滚轮。<br><br>"
            
            "<b>主要模式 (通过工具栏按钮切换):</b><br>"
            "- <b>绘制 [B]:</b> 添加蒙版区域 (前景)。<br>"
            "- <b>擦除 [E]:</b> 移除蒙版区域 (背景)。<br>"
            "- <b>GrabCut (自动蒙版菜单):</b> 通过框选和标记进行交互式分割。<br>"
            "- <b>Watershed (自动蒙版菜单):</b> 通过标记前景/背景进行分割。<br><br>"
            
            "<b>工具栏按钮 (部分):</b><br>"
            "- <b>撤销/重做 [Ctrl+Z/Y]:</b> 撤销或重做编辑步骤。<br>"
            "- <b>重置 [R]:</b> 恢复到初始蒙版状态。<br>"
            "- <b>自动修复 [A]:</b> 使用形态学优化蒙版。<br>"
            "- <b>自动蒙版 [D]:</b> 提供多种自动分割算法入口。<br>"
            "- <b>重置视图 [Home]:</b> 恢复默认缩放和平移。<br>"
            "- <b>笔刷大小 [ \[ / \] ]:</b> 调整工具大小。<br><br>"

            "<b>键盘快捷键:</b><br>"
            "<table>"
            "<tr><td>B</td><td>切换到绘制模式</td></tr>"
            "<tr><td>E</td><td>切换到擦除模式</td></tr>"
            "<tr><td>[</td><td>减小笔刷大小</td></tr>"
            "<tr><td>]</td><td>增大笔刷大小</td></tr>"
            "<tr><td>Ctrl+Z</td><td>撤销</td></tr>"
            "<tr><td>Ctrl+Y</td><td>重做</td></tr>"
            "<tr><td>R</td><td>重置蒙版</td></tr>"
            "<tr><td>A</td><td>自动修复蒙版</td></tr>"
            "<tr><td>D</td><td>打开自动蒙版菜单</td></tr>"
            "<tr><td>Home</td><td>重置视图</td></tr>"
            "<tr><td>Ctrl+S</td><td>保存并关闭</td></tr>"
            "<tr><td>Esc</td><td>取消并关闭</td></tr>"
            "</table><br>"
            
            "<i>提示: 更详细的功能说明请查看项目 `tools/README.md` 文件。</i>"
        )
        
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setWindowTitle("帮助与快捷键")
        msg_box.setTextFormat(Qt.TextFormat.RichText)
        msg_box.setText(help_text)
        msg_box.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
        msg_box.exec()

    def show_grabcut_buttons(self, show):
        # Keep Undo/Redo? Maybe disable them during GrabCut refine?
        self.btn_undo.setEnabled(not show) 
        self.btn_redo.setEnabled(not show)
        self.ws_tools_frame.setVisible(False) # Ensure WS tools are hidden
        self.gc_tools_frame.setVisible(show)  # Control frame visibility
        # Disable standard draw/erase buttons when GC is active
        self.btn_draw.setEnabled(not show)
        self.btn_erase.setEnabled(not show)
        if show:
             self.btn_gc_fg.setChecked(True) # Default to FG marker

    def show_watershed_buttons(self, show=True):
        """显示或隐藏 Watershed 相关的工具栏按钮。"""
        logging.debug(f"Setting Watershed buttons visibility to: {show}")
        # 直接控制按钮可见性，而不通过Frame
        self.btn_ws_fg.setVisible(show)
        self.btn_ws_bg.setVisible(show)
        self.btn_ws_run.setVisible(show)
        
        if show:
            logging.debug("Set Watershed buttons to visible")
            # 直接获取父工具栏并刷新
            self.toolbar.update()
            logging.debug("Toolbar updated")
            
            # 在下一个事件循环中，再次确保按钮可见性
            QtCore.QTimer.singleShot(100, lambda: (
                self.btn_ws_fg.setVisible(True),
                self.btn_ws_bg.setVisible(True),
                self.btn_ws_run.setVisible(True),
                logging.debug("Delayed WS buttons visibility refresh")
            ))

        self.gc_tools_frame.setVisible(False) # Ensure GC tools are hidden
        
        # Disable other conflicting modes/buttons
        self.btn_draw.setEnabled(not show)
        self.btn_erase.setEnabled(not show)
        self.btn_auto_fix.setEnabled(not show)
        self.btn_auto_mask.setEnabled(not show)
        self.btn_reset.setEnabled(not show)
        self.btn_undo.setEnabled(not show) 
        self.btn_redo.setEnabled(not show)
        
        if show:
            self.btn_ws_fg.setChecked(True) # Default to FG marker
            logging.debug("WS foreground marker button checked")

    def on_watershed_marker_selected(self, button):
        """Handles selection changes in the Watershed marker button group."""
        # Determine the mode based on which button was clicked
        # Using object comparison is safer than relying on IDs if IDs change
        mode = 'fg' if button == self.btn_ws_fg else 'bg'
        logging.debug(f"Watershed marker mode selected: {mode}")
        # Call the widget's method to update its internal state
        self.edit_widget.set_watershed_marker_mode(mode)

class AnimationPreviewDialog(QtWidgets.QDialog):
    """用于预览提取的帧序列动画效果的对话框。

    允许调整帧率和循环播放，便于检查提取的帧是否构成连贯动画序列。
    """
    def __init__(self, rois, parent=None):
        """初始化AnimationPreviewDialog。
        
        Args:
            rois (list[FrameROI]): 包含帧数据的FrameROI对象列表。
            parent (QWidget, optional): 父控件. Defaults to None.
        """
        super().__init__(parent)
        self.rois = rois
        self.setWindowTitle("动画预览")
        self.setMinimumSize(400, 400)
        
        # 动画控制参数
        self.current_frame = 0
        self.fps = 12  # 默认每秒12帧
        self.playing = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)
        
        # 布局
        layout = QtWidgets.QVBoxLayout(self)
        
        # 预览图像标签
        self.image_label = QtWidgets.QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(300, 300)
        layout.addWidget(self.image_label)
        
        # 控制区域
        control_layout = QtWidgets.QHBoxLayout()
        
        # 播放/暂停按钮
        self.play_btn = QtWidgets.QPushButton("播放")
        self.play_btn.clicked.connect(self.toggle_play)
        control_layout.addWidget(self.play_btn)
        
        # 帧率控制
        control_layout.addWidget(QtWidgets.QLabel("帧率:"))
        self.fps_spin = QtWidgets.QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(self.fps)
        self.fps_spin.valueChanged.connect(self.set_fps)
        control_layout.addWidget(self.fps_spin)
        
        # 帧计数显示
        self.frame_label = QtWidgets.QLabel(f"帧: 1/{len(self.rois)}")
        control_layout.addWidget(self.frame_label)
        
        # 关闭按钮
        close_btn = QtWidgets.QPushButton("关闭")
        close_btn.clicked.connect(self.reject)
        control_layout.addWidget(close_btn)
        
        layout.addLayout(control_layout)
        
        # 显示第一帧
        self.update_display()
        
    def set_fps(self, fps):
        """设置帧率并更新计时器。
        
        Args:
            fps (int): 新的帧率。
        """
        self.fps = fps
        if self.playing:
            self.timer.stop()
            self.timer.start(1000 // self.fps)
            
    def toggle_play(self):
        """切换播放/暂停状态。"""
        if self.playing:
            self.timer.stop()
            self.play_btn.setText("播放")
        else:
            self.timer.start(1000 // self.fps)
            self.play_btn.setText("暂停")
        self.playing = not self.playing
        
    def next_frame(self):
        """显示下一帧，并在到达最后一帧时循环。"""
        self.current_frame = (self.current_frame + 1) % len(self.rois)
        self.update_display()
        
    def update_display(self):
        """更新图像显示。"""
        if not self.rois:
            return
            
        roi = self.rois[self.current_frame]
        try:
            img_pil = Image.fromarray(roi.img)
            img_qt = ImageQt.ImageQt(img_pil)
            pixmap = QPixmap.fromImage(img_qt)
            scaled_pixmap = pixmap.scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
            # 更新帧计数
            self.frame_label.setText(f"帧: {self.current_frame+1}/{len(self.rois)}")
        except Exception as e:
            logging.exception(f"Error updating animation frame: {e}")
            
    def closeEvent(self, event):
        """关闭事件：确保停止计时器。"""
        self.timer.stop()
        super().closeEvent(event)
