"""
---------------------------------------------------------------
File name:                  main_window.py
Author:                     Ignorant-lu
Date created:               2025/04/15
Description:                Sprite Mask Editor工具的主窗口
----------------------------------------------------------------

Changed history:            
                            2025/04/15: 从sprite_mask_editor.py拆分为独立模块;
                            2025/04/15: 迁移SpriteMaskEditor类实现;
----
"""
import sys
import os
import numpy as np
from PyQt6 import QtWidgets, QtGui, QtCore
from PyQt6.QtCore import QSettings, QTimer, Qt, QSize
from PyQt6.QtGui import QIcon, QAction, QCursor, QPixmap, QImage
from PIL import Image, ImageQt
import cv2
import json
import logging
from PyQt6.QtWidgets import QTextEdit, QDockWidget

from .constants import APP_NAME, APP_VERSION, DEFAULT_FRAME_SIZE, DEFAULT_BRUSH_SIZE
from .widgets import ParamHelpLabel, ThumbListWidget, MaskEditWidget
from .dialogs import MaskEditDialog, AnimationPreviewDialog
from .mask_processor import MaskProcessor, sort_rois, filter_rois, render_filename
from .presets import PresetManager
from .roi import FrameROI

# Custom Logging Handler for Qt
class QtLogHandler(logging.Handler, QtCore.QObject):
    """自定义日志处理器，将日志记录发送到Qt信号。"""
    log_signal = QtCore.pyqtSignal(str)

    def __init__(self):
        logging.Handler.__init__(self)
        QtCore.QObject.__init__(self)

    def emit(self, record):
        """格式化日志记录并发出信号。"""
        try:
            msg = self.format(record)
            self.log_signal.emit(msg)
        except Exception:
            self.handleError(record)

class SpriteMaskEditor(QtWidgets.QMainWindow):
    """Sprite Mask 可视化编辑工具的主窗口类。

    提供了加载sprite sheet图像、调整参数生成蒙版和提取帧、
    预览和编辑帧、管理参数预设、导出结果等功能。
    """
    def __init__(self):
        """初始化SpriteMaskEditor主窗口。"""
        super().__init__()
        self.setWindowTitle('Sprite Mask 可视化操作台')
        self.setMinimumSize(1200, 800)
        
        # 数据初始化
        self.image = None
        self.img_np = None
        self.mask = None
        self.rois = []
        self._all_rois = [] # Store all rois before filtering/sorting
        self.current_idx = -1 # Initialize to -1
        self.processor = MaskProcessor()
        
        # 排序、筛选、命名相关
        self.sort_by = "idx" # Default to idx (extraction order)
        self.sort_reverse = False # Default to ascending for idx
        self.area_range = (self.processor.min_area, self.processor.max_area) # Initialize from processor defaults
        self.aspect_range = (0, 99) # Default aspect ratio range
        self.naming_template = "frame_{idx:02d}_{x}_{y}.png"
        
        # 参数更新节流
        self.param_update_timer = QTimer(self)
        self.param_update_timer.setSingleShot(True)
        self.param_update_timer.timeout.connect(self.delayed_param_update)
        
        # 预设管理器
        self.preset_manager = PresetManager(APP_NAME)
        
        # 设置初始化
        self.settings = QSettings(APP_NAME, APP_NAME)
        self.last_dir = "." # Initialize last_dir
        
        # 创建UI界面
        self._init_ui()
        
        # 连接信号
        self.connect_signals() # Call signal connection method
        
        # 加载保存的设置
        self._load_settings()
        
        # 根据加载的设置更新UI控件的初始值
        self._update_ui_from_settings()
        # --- Initialize Logging --- 
        self._init_logging()

    def _init_logging(self):
        """初始化日志系统。"""
        # Create QTextEdit for log output
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet("background-color:#f0f0f0; font-family: Consolas, Courier New, monospace;")

        # Create Dock Widget for the log output
        log_dock = QDockWidget("日志输出", self)
        log_dock.setWidget(self.log_output)
        log_dock.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea | Qt.DockWidgetArea.TopDockWidgetArea)
        log_dock.setObjectName("LogDockWidget") # Set object name for saving state
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, log_dock)
        # Make the dock widget initially visible but allow closing
        log_dock.setVisible(True) 
        log_dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetClosable | QDockWidget.DockWidgetFeature.DockWidgetMovable | QDockWidget.DockWidgetFeature.DockWidgetFloatable)

        # Setup logging handler
        log_handler = QtLogHandler()
        log_handler.log_signal.connect(self.append_log_message)

        # Configure logging format
        log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', datefmt='%H:%M:%S')
        log_handler.setFormatter(log_format)
        
        # File handler (recommended)
        try:
            log_file = f"{APP_NAME.lower()}.log"
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(log_format)
        except Exception as e:
            print(f"Error setting up file logger: {e}") 
            file_handler = None

        # Get root logger and add handlers
        logger = logging.getLogger() # Get root logger
        logger.setLevel(logging.DEBUG) # Set desired level to DEBUG
        logger.addHandler(log_handler)
        if file_handler:
            logger.addHandler(file_handler)

        # Log application start
        logging.info(f"{APP_NAME} v{APP_VERSION} started.")
        logging.info(f"Log file location: {os.path.abspath(log_file) if file_handler else 'Not available'}")

    def append_log_message(self, message):
        """槽函数：将日志消息追加到日志输出窗口。"""
        # Ensure this runs in the main thread if logging happens in other threads
        self.log_output.append(message)
        self.log_output.verticalScrollBar().setValue(self.log_output.verticalScrollBar().maximum()) # Auto-scroll

    def _init_ui(self):
        """初始化用户界面布局和控件。"""
        # 创建中央控件和主布局
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        
        # 顶部工具栏
        toolbar = QtWidgets.QToolBar("主工具栏")
        self.addToolBar(toolbar)
        
        # 加载图片按钮
        self.load_action = QAction(QIcon(":/icons/open.png"), "加载图片 (Ctrl+O)", self)
        toolbar.addAction(self.load_action)
        
        # 导出全部按钮
        self.export_action = QAction(QIcon(":/icons/export.png"), "导出全部 (Ctrl+S)", self)
        toolbar.addAction(self.export_action)
        
        # 批量操作按钮
        self.batch_export_btn = QtWidgets.QToolButton(self)
        self.batch_export_btn.setIcon(QIcon(":/icons/batch_export.png"))
        self.batch_export_btn.setText("批量导出")
        self.batch_export_btn.setToolTip("导出选中的帧")
        toolbar.addWidget(self.batch_export_btn)
        
        self.batch_tag_btn = QtWidgets.QToolButton(self)
        self.batch_tag_btn.setIcon(QIcon(":/icons/batch_tag.png"))
        self.batch_tag_btn.setText("批量标签")
        self.batch_tag_btn.setToolTip("为选中的帧批量设置标签")
        toolbar.addWidget(self.batch_tag_btn)
        
        self.batch_note_btn = QtWidgets.QToolButton(self)
        self.batch_note_btn.setIcon(QIcon(":/icons/batch_note.png"))
        self.batch_note_btn.setText("批量备注")
        self.batch_note_btn.setToolTip("为选中的帧批量设置备注")
        toolbar.addWidget(self.batch_note_btn)
        
        self.batch_import_btn = QtWidgets.QToolButton(self)
        self.batch_import_btn.setIcon(QIcon(":/icons/batch_import.png"))
        self.batch_import_btn.setText("批量导入")
        self.batch_import_btn.setToolTip("从JSON文件导入标签/备注")
        toolbar.addWidget(self.batch_import_btn)
        
        # 预设管理
        toolbar.addSeparator()
        
        preset_menu = QtWidgets.QMenu("参数预设", self)
        
        self.save_preset_action = QAction(QIcon(":/icons/save_preset.png"), "保存当前预设", self)
        preset_menu.addAction(self.save_preset_action)
        
        self.load_preset_action = QAction(QIcon(":/icons/load_preset.png"), "加载预设", self)
        preset_menu.addAction(self.load_preset_action)
        
        self.delete_preset_action = QAction(QIcon(":/icons/delete_preset.png"), "删除预设", self)
        preset_menu.addAction(self.delete_preset_action)
        
        preset_button = QtWidgets.QToolButton()
        preset_button.setIcon(QIcon(":/icons/preset.png"))
        preset_button.setText("预设")
        preset_button.setMenu(preset_menu)
        preset_button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        toolbar.addWidget(preset_button)
        
        # 动画预览按钮
        self.preview_action = QAction(QIcon(":/icons/play.png"), "动画预览 (Ctrl+P)", self)
        toolbar.addAction(self.preview_action)
        
        # 帮助按钮
        toolbar.addSeparator()
        self.help_action = QAction(QIcon(":/icons/help.png"), "帮助", self)
        toolbar.addAction(self.help_action)
        
        # 分割布局
        splitter = QtWidgets.QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # 左侧参数区
        params_widget = QtWidgets.QWidget()
        params_layout = QtWidgets.QVBoxLayout(params_widget)
        
        # 参数选项卡
        params_tabs = QtWidgets.QTabWidget()
        params_tabs.setTabPosition(QtWidgets.QTabWidget.TabPosition.North)
        
        # 基础参数选项卡
        basic_tab = QtWidgets.QWidget()
        basic_layout = QtWidgets.QFormLayout(basic_tab)
        
        self.thresh_slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.thresh_slider.setRange(1, 100)
        self.thresh_slider.setValue(self.processor.color_thresh)
        self.thresh_label = ParamHelpLabel(f"色差阈值: {self.processor.color_thresh}", 
                                           "色差阈值：像素与背景色的距离，大于此值视为前景。调高可排除更多接近背景的像素，但易丢失细节。")
        
        self.pad_spin = QtWidgets.QSpinBox()
        self.pad_spin.setRange(0, 20)
        self.pad_spin.setValue(self.processor.pad)
        self.pad_label = ParamHelpLabel("Pad:", "边界扩展像素数：提取轮廓区域时向外扩展的像素数，防止角色被裁剪。")
        
        self.kernel_spin = QtWidgets.QSpinBox()
        self.kernel_spin.setRange(1, 15)
        self.kernel_spin.setStepType(QtWidgets.QAbstractSpinBox.StepType.AdaptiveDecimalStepType)
        self.kernel_spin.setValue(self.processor.kernel_size)
        self.kernel_label = ParamHelpLabel("核大小:", "形态学操作核大小：影响闭/开运算的范围，调大可去除大面积杂点，调小保留细节。")
        
        basic_layout.addRow(self.thresh_label, self.thresh_slider)
        basic_layout.addRow(self.pad_label, self.pad_spin)
        basic_layout.addRow(self.kernel_label, self.kernel_spin)
        
        # 形态学参数
        self.close_iter_spin = QtWidgets.QSpinBox()
        self.close_iter_spin.setRange(0, 8)
        self.close_iter_spin.setValue(self.processor.close_iter)
        self.close_iter_label = ParamHelpLabel("闭运算次数:", "闭运算(先膨胀后腐蚀)次数：可填补mask小孔洞，调高可让mask更平滑。")
        
        self.open_iter_spin = QtWidgets.QSpinBox()
        self.open_iter_spin.setRange(0, 8)
        self.open_iter_spin.setValue(self.processor.open_iter)
        self.open_iter_label = ParamHelpLabel("开运算次数:", "开运算(先腐蚀后膨胀)次数：可去除mask小杂点，调高可净化边缘。")
        
        basic_layout.addRow(self.close_iter_label, self.close_iter_spin)
        basic_layout.addRow(self.open_iter_label, self.open_iter_spin)
        
        # 输出参数选项卡
        output_tab = QtWidgets.QWidget()
        output_layout = QtWidgets.QFormLayout(output_tab)
        
        self.max_extract_spin = QtWidgets.QSpinBox()
        self.max_extract_spin.setRange(1, 256)
        self.max_extract_spin.setValue(self.processor.max_extract)
        self.max_extract_label = ParamHelpLabel("最大帧数:", "每次最多提取的角色帧数。")
        
        self.out_width_spin = QtWidgets.QSpinBox()
        self.out_width_spin.setRange(8, 1024)
        self.out_width_spin.setValue(self.processor.out_width)
        self.out_width_label = ParamHelpLabel("输出宽度:", "导出帧的画布宽度，所有帧自动居中。")
        
        self.out_height_spin = QtWidgets.QSpinBox()
        self.out_height_spin.setRange(8, 1024)
        self.out_height_spin.setValue(self.processor.out_height)
        self.out_height_label = ParamHelpLabel("输出高度:", "导出帧的画布高度，所有帧自动居中。")
        
        output_layout.addRow(self.max_extract_label, self.max_extract_spin)
        output_layout.addRow(self.out_width_label, self.out_width_spin)
        output_layout.addRow(self.out_height_label, self.out_height_spin)
        
        # 排序筛选选项卡
        filter_tab = QtWidgets.QWidget()
        filter_layout = QtWidgets.QFormLayout(filter_tab)
        
        # 排序方式
        self.sort_combo = QtWidgets.QComboBox()
        self.sort_combo.addItems(["默认 (提取顺序)", "轮廓面积 (area)", "左坐标 (x)", "上坐标 (y)", "宽度 (w)", "高度 (h)", "长宽比 (aspect_ratio)", "索引号 (idx)"])
        self.sort_combo.setToolTip("选择帧列表的排序依据。'默认'表示按提取顺序。")
        filter_layout.addRow("排序方式:", self.sort_combo)
        
        # 排序顺序
        self.sort_order_combo = QtWidgets.QComboBox()
        self.sort_order_combo.addItems(["降序 (大->小)", "升序 (小->大)"])
        self.sort_order_combo.setToolTip("选择排序方向。默认排序无效。")
        self.sort_order_combo.setEnabled(False)
        filter_layout.addRow("排序顺序:", self.sort_order_combo)
        
        # 面积筛选
        self.area_min_spin = QtWidgets.QSpinBox()
        self.area_min_spin.setRange(0, 9999999)
        self.area_min_spin.setValue(self.processor.min_area)
        
        self.area_max_spin = QtWidgets.QSpinBox()
        self.area_max_spin.setRange(0, 9999999)
        self.area_max_spin.setValue(self.processor.max_area)
        
        area_box = QtWidgets.QHBoxLayout()
        area_box.addWidget(QtWidgets.QLabel("最小:"))
        area_box.addWidget(self.area_min_spin)
        area_box.addWidget(QtWidgets.QLabel("最大:"))
        area_box.addWidget(self.area_max_spin)
        
        area_filter_widget = QtWidgets.QWidget()
        area_filter_widget.setLayout(area_box)
        filter_layout.addRow("面积筛选:", area_filter_widget)
        
        # 长宽比筛选
        self.aspect_min_spin = QtWidgets.QDoubleSpinBox()
        self.aspect_min_spin.setRange(0, 99)
        self.aspect_min_spin.setDecimals(2)
        self.aspect_min_spin.setValue(self.aspect_range[0])
        
        self.aspect_max_spin = QtWidgets.QDoubleSpinBox()
        self.aspect_max_spin.setRange(0, 99)
        self.aspect_max_spin.setDecimals(2)
        self.aspect_max_spin.setValue(self.aspect_range[1])
        
        aspect_box = QtWidgets.QHBoxLayout()
        aspect_box.addWidget(QtWidgets.QLabel("最小:"))
        aspect_box.addWidget(self.aspect_min_spin)
        aspect_box.addWidget(QtWidgets.QLabel("最大:"))
        aspect_box.addWidget(self.aspect_max_spin)
        
        aspect_filter_widget = QtWidgets.QWidget()
        aspect_filter_widget.setLayout(aspect_box)
        filter_layout.addRow("长宽比筛选:", aspect_filter_widget)
        
        # 命名模板 - 使用 QHBoxLayout 包含输入框和按钮
        naming_layout = QtWidgets.QHBoxLayout()
        self.naming_input = QtWidgets.QLineEdit(self.naming_template)
        self.naming_input.setToolTip("自定义导出文件名模板。使用 [占位符] 或 [占位符:格式]，如 [索引:03d]、[标签]。")
        naming_layout.addWidget(self.naming_input, 1) # Stretch factor 1

        # 插入占位符按钮
        self.insert_placeholder_btn = QtWidgets.QToolButton()
        self.insert_placeholder_btn.setText("插入")
        self.insert_placeholder_btn.setIcon(QIcon(":/icons/add.png"))
        self.insert_placeholder_btn.setToolTip("插入占位符到当前光标位置")
        insert_menu = QtWidgets.QMenu(self)
        placeholders = {
            "[索引]": "帧的序号",
            "[X]": "帧左上角X坐标",
            "[Y]": "帧左上角Y坐标",
            "[宽]": "帧宽度",
            "[高]": "帧高度",
            "[面积]": "帧轮廓面积",
            "[长宽比]": "帧长宽比",
            "[标签]": "用户自定义标签",
            "[备注]": "用户自定义备注",
        }
        for p_text, p_tooltip in placeholders.items():
            action = QAction(p_text, self)
            action.setToolTip(p_tooltip)
            action.triggered.connect(lambda checked=False, text=p_text: self.insert_placeholder(text))
            insert_menu.addAction(action)
        self.insert_placeholder_btn.setMenu(insert_menu)
        self.insert_placeholder_btn.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        naming_layout.addWidget(self.insert_placeholder_btn)

        # 预设模板按钮
        self.preset_template_btn = QtWidgets.QToolButton()
        self.preset_template_btn.setText("预设")
        self.preset_template_btn.setIcon(QIcon(":/icons/template.png"))
        self.preset_template_btn.setToolTip("选择预设命名模板")
        preset_menu = QtWidgets.QMenu(self)
        preset_templates = {
            "帧_[索引:03d]": "Frame_001, Frame_002, ...",
            "[标签]_[索引]": "Walk_1, Walk_2, ...",
            "Sprite_[X]_[Y]": "Sprite_128_256, ...",
            "[索引]": "1, 2, 3, ..."
        }
        for t_text, t_tooltip in preset_templates.items():
            action = QAction(t_text, self)
            action.setToolTip(t_tooltip)
            action.triggered.connect(lambda checked=False, text=t_text: self.apply_preset_template(text))
            preset_menu.addAction(action)
        self.preset_template_btn.setMenu(preset_menu)
        self.preset_template_btn.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        naming_layout.addWidget(self.preset_template_btn)

        naming_widget = QtWidgets.QWidget()
        naming_widget.setLayout(naming_layout)
        filter_layout.addRow("命名模板:", naming_widget)
        
        self.naming_preview = QtWidgets.QLabel("N/A")
        filter_layout.addRow("预览:", self.naming_preview)
        
        # 添加选项卡
        params_tabs.addTab(basic_tab, "基础参数")
        params_tabs.addTab(output_tab, "输出设置")
        params_tabs.addTab(filter_tab, "排序/筛选")
        
        params_layout.addWidget(params_tabs)
        
        # 预览区域
        preview_widget = QtWidgets.QWidget()
        preview_layout = QtWidgets.QVBoxLayout(preview_widget)
        
        # 预览标签和操作区
        preview_info_layout = QtWidgets.QHBoxLayout()
        self.preview_info_label = QtWidgets.QLabel("未加载图片")
        preview_info_layout.addWidget(self.preview_info_label)
        
        self.edit_mask_btn = QtWidgets.QPushButton("编辑Mask (Ctrl+E)")
        self.edit_mask_btn.setIcon(QIcon(":/icons/edit.png"))
        self.edit_mask_btn.setEnabled(False)
        preview_info_layout.addWidget(self.edit_mask_btn)
        
        preview_layout.addLayout(preview_info_layout)
        
        # 原图和Mask预览
        previews_layout = QtWidgets.QHBoxLayout()
        
        # 原图预览区
        img_group = QtWidgets.QGroupBox("原图")
        img_layout = QtWidgets.QVBoxLayout(img_group)
        self.img_label = QtWidgets.QLabel()
        self.img_label.setMinimumSize(256, 256)
        self.img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_label.setStyleSheet('background:#eee;border-radius:8px;')
        img_layout.addWidget(self.img_label, 1, Qt.AlignmentFlag.AlignCenter)
        
        # Mask预览区
        mask_group = QtWidgets.QGroupBox("Mask预览")
        mask_layout = QtWidgets.QVBoxLayout(mask_group)
        self.mask_label = QtWidgets.QLabel()
        self.mask_label.setMinimumSize(256, 256)
        self.mask_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.mask_label.setStyleSheet('background:#eee;border-radius:8px;')
        mask_layout.addWidget(self.mask_label, 1, Qt.AlignmentFlag.AlignCenter)
        
        previews_layout.addWidget(img_group)
        previews_layout.addWidget(mask_group)
        preview_layout.addLayout(previews_layout)
        
        # 标签与备注输入区
        metadata_layout = QtWidgets.QHBoxLayout()
        
        self.tag_edit = QtWidgets.QLineEdit()
        self.tag_edit.setPlaceholderText("标签")
        self.tag_edit.setEnabled(False)
        self.note_edit = QtWidgets.QLineEdit()
        self.note_edit.setPlaceholderText("备注")
        self.note_edit.setEnabled(False)
        metadata_layout.addWidget(QtWidgets.QLabel("标签:"))
        metadata_layout.addWidget(self.tag_edit)
        metadata_layout.addWidget(QtWidgets.QLabel("备注:"))
        metadata_layout.addWidget(self.note_edit)
        
        preview_layout.addLayout(metadata_layout)
        
        # 添加左右区域到分割器
        splitter.addWidget(params_widget)
        splitter.addWidget(preview_widget)
        
        # 设置初始分割比例
        splitter.setSizes([350, 850])
        
        # 缩略图列表
        thumb_group = QtWidgets.QGroupBox("帧缩略图浏览")
        thumb_group_layout = QtWidgets.QVBoxLayout(thumb_group)
        self.thumb_list = ThumbListWidget(self)
        thumb_group_layout.addWidget(self.thumb_list)
        main_layout.addWidget(thumb_group)
        
        # 状态栏
        self.statusBar().showMessage("就绪。请加载图片开始操作。")

    def connect_signals(self):
        """连接所有UI控件的信号到槽函数。"""
        # 文件菜单
        self.load_action.triggered.connect(self.load_image)
        self.export_action.triggered.connect(self.export_all)
        
        # 批量操作按钮
        self.batch_export_btn.clicked.connect(self.batch_export)
        self.batch_tag_btn.clicked.connect(self.batch_set_tag)
        self.batch_note_btn.clicked.connect(self.batch_set_note)
        self.batch_import_btn.clicked.connect(self.batch_import)
        
        # 预设菜单
        self.save_preset_action.triggered.connect(self.save_preset)
        self.load_preset_action.triggered.connect(self.load_preset)
        self.delete_preset_action.triggered.connect(self.delete_preset)
        
        # 工具栏动作
        self.preview_action.triggered.connect(self.show_animation_preview)
        self.help_action.triggered.connect(self.show_help)
        
        # 参数控件 - 基础
        self.thresh_slider.valueChanged.connect(lambda value: (
            self.thresh_label.setText(f"色差阈值: {value}"),
            self.param_update_timer.start(300)
        ))
        self.pad_spin.valueChanged.connect(lambda: self.on_param_change(affects='roi'))
        self.kernel_spin.valueChanged.connect(lambda: self.param_update_timer.start(300))
        self.close_iter_spin.valueChanged.connect(lambda: self.param_update_timer.start(300))
        self.open_iter_spin.valueChanged.connect(lambda: self.param_update_timer.start(300))
        
        # 参数控件 - 输出
        self.max_extract_spin.valueChanged.connect(lambda: self.on_param_change(affects='roi'))
        self.out_width_spin.valueChanged.connect(lambda: self.on_param_change(affects='roi'))
        self.out_height_spin.valueChanged.connect(lambda: self.on_param_change(affects='roi'))
        
        # 参数控件 - 排序/筛选
        self.sort_combo.currentIndexChanged.connect(self.on_sort_change)
        self.sort_order_combo.currentIndexChanged.connect(self.on_sort_order_change)
        self.area_min_spin.valueChanged.connect(self.on_filter_change)
        self.area_max_spin.valueChanged.connect(self.on_filter_change)
        self.aspect_min_spin.valueChanged.connect(self.on_filter_change)
        self.aspect_max_spin.valueChanged.connect(self.on_filter_change)
        self.naming_input.textChanged.connect(self.on_naming_change)
        
        # 预览区
        self.edit_mask_btn.clicked.connect(self.on_edit_mask)
        self.tag_edit.editingFinished.connect(self.on_tag_changed)
        self.note_edit.editingFinished.connect(self.on_note_changed)
        
        # 缩略图列表
        self.thumb_list.frame_selected.connect(self.on_frame_select)
        self.thumb_list.selection_changed.connect(self.on_selection_change)
        # Connect custom signals from ThumbListWidget
        self.thumb_list.edit_mask_requested.connect(self.on_thumb_edit_mask)
        self.thumb_list.batch_export_requested.connect(self.on_thumb_batch_export)
        self.thumb_list.batch_tag_requested.connect(self.on_thumb_batch_tag)
        self.thumb_list.batch_note_requested.connect(self.on_thumb_batch_note)

    def on_sort_change(self):
        """处理排序方式下拉框变化。"""
        idx = self.sort_combo.currentIndex()
        bys = ["idx", "area", "x", "y", "w", "h", "aspect_ratio", "idx"]
        selected_by = bys[idx]
        self.sort_by = selected_by

        if selected_by == "idx":
            self.sort_reverse = False
            self.sort_order_combo.setEnabled(False)
            self.sort_order_combo.blockSignals(True)
            self.sort_order_combo.setCurrentIndex(1)
            self.sort_order_combo.blockSignals(False)
        else:
            self.sort_order_combo.setEnabled(True)
            self.sort_reverse = self.sort_order_combo.currentIndex() == 0

        if self.rois:
            self.statusBar().showMessage("正在应用排序...", 1000)
            self.refresh_sort_filter()

    def on_sort_order_change(self):
        """处理排序顺序下拉框变化。"""
        if self.sort_by == "idx":
             return
             
        self.sort_reverse = self.sort_order_combo.currentIndex() == 0
        if self.rois:
            self.statusBar().showMessage("正在应用排序...", 1000)
            self.refresh_sort_filter()

    def on_filter_change(self):
        """处理筛选参数（面积、长宽比）变化。"""
        self.area_range = (self.area_min_spin.value(), self.area_max_spin.value())
        self.aspect_range = (self.aspect_min_spin.value(), self.aspect_max_spin.value())
        
        if self.rois:
            self.statusBar().showMessage("正在筛选...", 1000)
            self.refresh_sort_filter()

    def on_naming_change(self):
        """处理文件命名模板输入框变化，更新预览。"""
        self.naming_template = self.naming_input.text()
        
        if self.rois:
            try:
                # Use the first selected frame for preview if available, otherwise the first frame
                preview_idx = self.current_idx if 0 <= self.current_idx < len(self.rois) else 0
                if 0 <= preview_idx < len(self.rois):
                     # Use the updated render_filename function
                     name = render_filename(self.naming_template, self.rois[preview_idx])
                else:
                     name = "无可用帧预览"
            except Exception as e:
                # Use logging for the error
                logging.warning(f"生成命名预览时出错: {e}", exc_info=False)
                name = f"模板错误"
            self.naming_preview.setText(name)
        else:
            self.naming_preview.setText("N/A")
            
    def on_frame_select(self, idx):
        """槽函数：响应缩略图列表的frame_selected信号，更新当前选中帧和预览。"""
        if 0 <= idx < len(self.rois):
            self.current_idx = idx
        self.update_preview()
        
    def on_selection_change(self, selected_indices):
        """处理缩略图选择变化"""
        has_selection = len(selected_indices) > 0
        self.batch_export_btn.setEnabled(has_selection)
        self.batch_tag_btn.setEnabled(has_selection)
        self.batch_note_btn.setEnabled(has_selection)
        
        if len(selected_indices) > 1:
            self.preview_info_label.setText(f"已选择 {len(selected_indices)} 帧")
            self.edit_mask_btn.setEnabled(False)
            self.tag_edit.setEnabled(False)
            self.note_edit.setEnabled(False)
        elif len(selected_indices) == 1:
            idx = list(selected_indices)[0]
            if self.current_idx != idx:
                self.current_idx = idx
            self.update_preview()
        else:
            self.current_idx = -1
            self.update_preview()
            
    def delayed_param_update(self):
        """延迟的参数更新，用于防抖。仅处理影响 Mask 生成的参数。"""
        self.processor.color_thresh = self.thresh_slider.value()
        self.processor.kernel_size = self.kernel_spin.value()
        self.processor.close_iter = self.close_iter_spin.value()
        self.processor.open_iter = self.open_iter_spin.value()
        
        if self.img_np is not None:
            self.statusBar().showMessage("正在更新蒙版和区域...", 1000)
            self.refresh_mask_and_rois()
            
    def on_param_change(self, affects='unknown'):
        """处理影响 ROI 提取或排序/筛选的参数变化。"""
        self.processor.pad = self.pad_spin.value()
        self.processor.max_extract = self.max_extract_spin.value()
        self.processor.out_width = self.out_width_spin.value()
        self.processor.out_height = self.out_height_spin.value()
        
        if self.img_np is not None and self.mask is not None:
            self.statusBar().showMessage("正在更新区域...", 1000)
            self._extract_and_update_rois()
            
    def update_preview(self):
        """更新右侧预览区域（原图、Mask、标签、备注、信息）。"""
        if self.img_np is None or not self.rois or self.current_idx < 0 or self.current_idx >= len(self.rois):
            self.img_label.clear()
            self.img_label.setText("无预览")
            self.mask_label.clear()
            self.mask_label.setText("无预览")
            self.preview_info_label.setText("请加载图片或调整参数")
            self.tag_edit.setText("")
            self.tag_edit.setEnabled(False)
            self.note_edit.setText("")
            self.note_edit.setEnabled(False)
            self.edit_mask_btn.setEnabled(False)
            self.naming_preview.setText("N/A")
            return

        try:
            roi = self.rois[self.current_idx]
        except IndexError:
            logging.error(f"current_idx {self.current_idx} out of bounds for rois list (len: {len(self.rois)})")
            self.current_idx = -1
            self.update_preview()
            return

        try:
            original_roi_pixels = self.img_np[roi.y:roi.y+roi.h, roi.x:roi.x+roi.w].copy()
            if original_roi_pixels.size == 0:
                raise ValueError("Extracted original ROI pixels are empty.")
            img_qt = ImageQt.ImageQt(Image.fromarray(original_roi_pixels))
            pix = QPixmap.fromImage(img_qt)
            scaled_pix = pix.scaled(
                self.img_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.img_label.setPixmap(scaled_pix)
        except Exception as e:
            logging.exception(f"Error updating image preview for ROI #{roi.idx}: {e}")
            self.img_label.setText("原图错误")

        try:
            mask_data = roi.mask
            if mask_data is not None and mask_data.size > 0:
                mask_rgb = cv2.cvtColor(mask_data, cv2.COLOR_GRAY2RGB)
                mask_img = Image.fromarray(mask_rgb.astype(np.uint8))
                mask_qt = ImageQt.ImageQt(mask_img)
                mask_pix = QPixmap.fromImage(mask_qt)
                scaled_mask_pix = mask_pix.scaled(
                    self.mask_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.mask_label.setPixmap(scaled_mask_pix)
            else:
                self.mask_label.setText("无Mask")
                self.mask_label.setPixmap(QPixmap())
        except Exception as e:
            logging.exception(f"Error updating mask preview: {e}")
            self.mask_label.setText("Mask错误")
            self.mask_label.setPixmap(QPixmap())

        self.tag_edit.setText(roi.tag)
        self.tag_edit.setEnabled(True)
        self.note_edit.setText(roi.note)
        self.note_edit.setEnabled(True)

        self.preview_info_label.setText(f"帧 #{roi.idx} | 位置:({roi.x},{roi.y}) | 大小:{roi.w}x{roi.h} | 面积:{roi.area:.1f} | 长宽比:{roi.aspect_ratio:.2f}")

        self.edit_mask_btn.setEnabled(True)
        
        self.on_naming_change()

    def on_edit_mask(self):
        """槽函数：响应编辑Mask按钮点击。"""
        if not self.rois or self.current_idx < 0 or self.current_idx >= len(self.rois):
            return
            
        roi = self.rois[self.current_idx]
        
        original_img_segment = self.img_np[roi.y:roi.y+roi.h, roi.x:roi.x+roi.w].copy()
        current_mask = roi.get_current_mask()
        
        if current_mask.shape[:2] != original_img_segment.shape[:2]:
            current_mask = cv2.resize(current_mask, 
                                      (original_img_segment.shape[1], original_img_segment.shape[0]), 
                                      interpolation=cv2.INTER_NEAREST)

        dialog = MaskEditDialog(self, original_img_segment, current_mask)
        
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            edited_mask = dialog.get_mask()
            
            roi.mask = edited_mask
            roi.add_mask_to_history(edited_mask)

            try:
                if edited_mask.shape[:2] != roi.img.shape[:2]:
                    alpha_mask = cv2.resize(edited_mask, (roi.img.shape[1], roi.img.shape[0]), interpolation=cv2.INTER_NEAREST)
                else:
                    alpha_mask = edited_mask
                roi.img[..., 3] = alpha_mask
            except Exception as e:
                logging.exception(f"Error applying edited mask to ROI image alpha: {e}")

            self.update_preview()

            if 0 <= self.current_idx < len(self.thumb_list.thumb_labels):
                try:
                    qimg = ImageQt.ImageQt(Image.fromarray(roi.img))
                    thumb = QtGui.QPixmap.fromImage(qimg)
                    thumb = thumb.scaled(80, 80, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                    self.thumb_list.thumb_labels[self.current_idx].setPixmap(thumb)
                    self.thumb_list.update_selection_visuals()
                except Exception as e:
                    logging.exception(f"Error updating thumbnail after mask edit: {e}")
            
            self.statusBar().showMessage(f"帧 #{roi.idx} 的 Mask 已更新", 3000)

    def on_tag_changed(self):
        """槽函数：当标签输入框完成编辑时调用。"""
        if self.rois and 0 <= self.current_idx < len(self.rois):
            self.rois[self.current_idx].tag = self.tag_edit.text()
            self.on_naming_change()

    def on_note_changed(self):
        """槽函数：当备注输入框完成编辑时调用。"""
        if self.rois and 0 <= self.current_idx < len(self.rois):
            self.rois[self.current_idx].note = self.note_edit.text()

    def export_all(self):
        """导出当前筛选和排序后的所有帧。"""
        self._export_rois(self.rois, "导出全部帧")

    def batch_export(self):
        """导出选中的帧。"""
        selected_indices = self.thumb_list.selected_indices
        if not selected_indices:
            QtWidgets.QMessageBox.warning(self, "无选中帧", "请先在下方缩略图列表中选择要导出的帧。")
            return
        rois_to_export = [self.rois[i] for i in selected_indices if 0 <= i < len(self.rois)]
        self._export_rois(rois_to_export, "导出选中帧")

    def _export_rois(self, rois_list, title="导出帧"):
        """内部通用导出逻辑。"""
        if not rois_list:
            QtWidgets.QMessageBox.warning(self, "无帧可导出", f"没有找到可以导出的帧。")
            return
            
        out_dir = QtWidgets.QFileDialog.getExistingDirectory(self, title, self.last_dir)
        if not out_dir:
            return
            
        self.last_dir = out_dir
        
        progress = QtWidgets.QProgressDialog(f"正在导出 {len(rois_list)} 帧...", "取消", 0, len(rois_list), self)
        progress.setWindowTitle("导出进度")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        
        exported = 0
        errors = 0
        for i, roi in enumerate(rois_list):
            progress.setValue(i)
            if progress.wasCanceled():
                break
            
            try:
                out_pil = Image.fromarray(roi.img)
                try:
                    name = render_filename(self.naming_template, roi)
                except Exception as e:
                    logging.warning(f"Filename template error for ROI {roi.idx}: {e}, using default.")
                    name = f"frame_{roi.idx:02d}.png"
                    
                if not name.lower().endswith(".png"):
                    name += ".png"
                    
                out_path = os.path.join(out_dir, name)
                out_pil.save(out_path)
                exported += 1
            except Exception as e:
                errors += 1
                logging.error(f"Error exporting ROI #{roi.idx} ('{name}'): {str(e)}")
                self.statusBar().showMessage(f"导出第 {i+1} 帧时出错: {str(e)}", 3000)
            
        progress.setValue(len(rois_list))
        
        message = f"已成功导出 {exported} 帧。"
        if errors > 0:
            message += f"\n{errors} 帧导出失败，请查看控制台日志。"
        QtWidgets.QMessageBox.information(self, "导出完成", message)
        self.statusBar().showMessage(f"导出完成: {exported} 成功, {errors} 失败", 5000)
        
    def batch_set_tag(self):
        """为所有选中的帧设置相同标签。"""
        selected = self.thumb_list.selected_indices
        if not selected:
            return
            
        tag, ok = QtWidgets.QInputDialog.getText(self, "批量标签", "为选中的帧输入标签：")
        if ok:
            updated_count = 0
            for idx in selected:
                if 0 <= idx < len(self.rois):
                    self.rois[idx].tag = tag
                    updated_count += 1
                    
            if self.current_idx in selected:
                self.update_preview()
            self.statusBar().showMessage(f"已为 {updated_count} 帧设置标签: {tag}", 3000)
                
    def batch_set_note(self):
        """为所有选中的帧设置相同备注。"""
        selected = self.thumb_list.selected_indices
        if not selected:
            return
            
        note, ok = QtWidgets.QInputDialog.getText(self, "批量备注", "为选中的帧输入备注：")
        if ok:
            updated_count = 0
            for idx in selected:
                if 0 <= idx < len(self.rois):
                    self.rois[idx].note = note
                    updated_count += 1
                    
            if self.current_idx in selected:
                self.update_preview()
            self.statusBar().showMessage(f"已为 {updated_count} 帧设置备注", 3000)

    def batch_import(self):
        """从JSON文件导入标签和备注。JSON应为对象列表，每个对象包含 'idx', 'tag', 'note'。"""
        file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择导入文件", self.last_dir, "JSON Files (*.json)"
        )
        if not file:
            return
            
        self.last_dir = os.path.dirname(file)
        
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            if not isinstance(data, list):
                raise ValueError("JSON data must be a list of objects.")
            
            updated = 0
            roi_map = {roi.idx: roi for roi in self._all_rois}
            for item in data:
                if not isinstance(item, dict): continue
                idx = item.get("idx")
                if idx is not None and idx in roi_map:
                    roi = roi_map[idx]
                    if "tag" in item:
                        roi.tag = str(item["tag"])
                    if "note" in item:
                        roi.note = str(item["note"])
                    updated += 1
                    
            if updated > 0:
                self.update_preview()
                self.statusBar().showMessage(f"从文件更新了 {updated} 帧的标签/备注", 3000)
            else:
                self.statusBar().showMessage(f"未找到匹配的帧索引进行更新", 3000)
                
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "导入失败", f"无法导入数据: {str(e)}")
            self.statusBar().showMessage(f"导入失败: {e}", 5000)

    def on_thumb_edit_mask(self):
        """槽函数：处理缩略图右键菜单的"编辑Mask"动作。"""
        # 因为右键菜单的 contextMenuEvent 会确保点击的帧是当前选中的唯一帧
        self.on_edit_mask()
        
    def on_thumb_batch_export(self):
        """槽函数：处理缩略图右键菜单的"导出选中帧"动作。"""
        self.batch_export()

    def on_thumb_batch_tag(self):
        """槽函数：处理缩略图右键菜单的"批量设置标签"动作。"""
        self.batch_set_tag()

    def on_thumb_batch_note(self):
        """槽函数：处理缩略图右键菜单的"批量设置备注"动作。"""
        self.batch_set_note()

    def save_preset(self):
        """保存当前参数为预设。"""
        name, ok = QtWidgets.QInputDialog.getText(self, "保存预设", "输入预设名称:")
        if ok and name:
            preset_data = {
                "processor": self.processor.get_params(),
                "sort_by": self.sort_by,
                "sort_reverse": self.sort_reverse,
                "area_range": [self.area_min_spin.value(), self.area_max_spin.value()],
                "aspect_range": [self.aspect_min_spin.value(), self.aspect_max_spin.value()],
                "naming_template": self.naming_template
            }
            
            if self.preset_manager.save_preset(name, preset_data):
                self.statusBar().showMessage(f"预设 '{name}' 已保存", 3000)
            else:
                QtWidgets.QMessageBox.warning(self, "保存失败", f"无法保存预设 '{name}'")

    def load_preset(self):
        """加载选定的预设。"""
        presets = self.preset_manager.get_presets_list()
        if not presets:
            QtWidgets.QMessageBox.information(self, "无预设", "没有找到保存的预设")
            return
            
        preset_name, ok = QtWidgets.QInputDialog.getItem(self, "加载预设", "选择预设:", presets, 0, False)
        if ok and preset_name:
            preset_data = self.preset_manager.load_preset(preset_name)
            if preset_data:
                try:
                    self.processor.set_params(preset_data["processor"])
                    
                    self.sort_by = preset_data["sort_by"]
                    sort_map = {"idx": 0, "area": 1, "x": 2, "y": 3, "w": 4, "h": 5, "aspect_ratio": 6}
                    if self.sort_by in sort_map:
                        self.sort_combo.setCurrentIndex(sort_map[self.sort_by])
                    else:
                        self.sort_combo.setCurrentIndex(0)
                        
                    self.sort_reverse = preset_data["sort_reverse"]
                    self.sort_order_combo.setCurrentIndex(0 if self.sort_reverse else 1)
                    
                    self.area_min_spin.setValue(preset_data["area_range"][0])
                    self.area_max_spin.setValue(preset_data["area_range"][1])
                    self.area_range = tuple(preset_data["area_range"])
                    
                    self.aspect_min_spin.setValue(preset_data["aspect_range"][0])
                    self.aspect_max_spin.setValue(preset_data["aspect_range"][1])
                    self.aspect_range = tuple(preset_data["aspect_range"])
                    
                    self.naming_template = preset_data["naming_template"]
                    self.naming_input.setText(self.naming_template)
                    
                    self.thresh_slider.setValue(self.processor.color_thresh)
                    self.pad_spin.setValue(self.processor.pad)
                    self.kernel_spin.setValue(self.processor.kernel_size)
                    self.max_extract_spin.setValue(self.processor.max_extract)
                    self.close_iter_spin.setValue(self.processor.close_iter)
                    self.open_iter_spin.setValue(self.processor.open_iter)
                    self.out_width_spin.setValue(self.processor.out_width)
                    self.out_height_spin.setValue(self.processor.out_height)
                    
                    if self.img_np is not None:
                        self.refresh_mask_and_rois()
                        
                    self.statusBar().showMessage(f"预设 '{preset_name}' 已加载", 3000)
                except Exception as e:
                    QtWidgets.QMessageBox.warning(self, "加载失败", f"应用预设 '{preset_name}' 时出错: {e}")
                    self.statusBar().showMessage(f"加载预设 '{preset_name}' 出错", 5000)
            else:
                QtWidgets.QMessageBox.warning(self, "加载失败", f"无法加载预设 '{preset_name}'")
                
    def delete_preset(self):
        """删除选定的预设。"""
        presets = self.preset_manager.get_presets_list()
        if not presets:
            QtWidgets.QMessageBox.information(self, "无预设", "没有找到保存的预设")
            return
            
        preset_name, ok = QtWidgets.QInputDialog.getItem(self, "删除预设", "选择要删除的预设:", presets, 0, False)
        if ok and preset_name:
            confirm = QtWidgets.QMessageBox.question(self, "确认删除", 
                                                  f"确定要删除预设 '{preset_name}' 吗？此操作不可撤销。",
                                                  QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No)
            if confirm == QtWidgets.QMessageBox.StandardButton.Yes:
                if self.preset_manager.delete_preset(preset_name):
                    self.statusBar().showMessage(f"预设 '{preset_name}' 已删除", 3000)
                else:
                    QtWidgets.QMessageBox.warning(self, "删除失败", f"无法删除预设 '{preset_name}'")

    def show_animation_preview(self):
        """显示动画预览对话框。"""
        if not self.rois:
            QtWidgets.QMessageBox.information(self, "无帧预览", "请先加载图片并提取帧。")
            return
        dialog = AnimationPreviewDialog(self.rois, parent=self)
        dialog.exec()

    def show_help(self):
        """显示帮助信息对话框。"""
        help_text = """
        <h2>Sprite Mask 可视化操作台 - 帮助</h2>
        <p><b>基本操作：</b></p>
        <ul>
            <li><b>加载图片 (Ctrl+O):</b> 打开 sprite sheet 图片文件。</li>
            <li><b>调整参数:</b> 在左侧选项卡中修改参数以优化蒙版生成和帧提取。</li>
            <li><b>查看帧:</b> 提取的帧会显示在底部的缩略图列表中。</li>
            <li><b>选择帧:</b> 点击缩略图选择单帧进行预览和编辑。</li>
            <li><b>多选帧:</b> 按住 Ctrl 点击可多选，按住 Shift 点击可范围选择。</li>
            <li><b>右键菜单:</b> 在缩略图上右键可进行批量操作。</li>
            <li><b>编辑Mask (Ctrl+E):</b> 选中单帧后，点击按钮或按快捷键进入手动编辑蒙版模式。</li>
            <li><b>动画预览 (Ctrl+P):</b> 查看当前筛选和排序后的帧序列动画效果。</li>
            <li><b>导出 (Ctrl+S/按钮):</b> 导出全部或选中的帧为带透明通道的 PNG 图片。</li>
        </ul>
        
        <p><b>参数说明 (悬停在标签上可看详细提示):</b></p>
        <ul>
            <li><b>基础参数:</b> 控制基于背景色的自动蒙版生成算法。</li>
            <li><b>输出设置:</b> 控制提取的帧数量和输出画布大小。</li>
            <li><b>排序/筛选:</b> 控制帧的排序方式和基于面积/长宽比的筛选。</li>
            <li><b>命名模板:</b> 自定义导出文件的命名规则。使用友好占位符如 `[索引]`, `[标签]` 或 `[索引:03d]`。</li>
        </ul>
        
        <p><b>蒙版编辑快捷键 (编辑对话框内):</b></p>
        <ul>
            <li>B: 绘制模式</li>
            <li>E: 擦除模式</li>
            <li>[: 减小笔刷</li>
            <li>]: 增大笔刷</li>
            <li>Ctrl+Z: 撤销</li>
            <li>Ctrl+Y: 重做</li>
            <li>R: 重置蒙版 (清空)</li>
            <li>A: <b>自动修复:</b> 弹出对话框调整形态学参数以去噪点/填洞。
                <ul>
                    <li>开运算(去噪点): 核大小/迭代次数</li>
                    <li>闭运算(填洞): 核大小/迭代次数</li>
                </ul>
            </li>
            <li>D: <b>自动蒙版:</b> 弹出算法选择菜单或参数对话框（当前为Canny）。
                 <ul>
                    <li><b>Canny参数:</b></li>
                    <li>低阈值/高阈值: 控制边缘检测敏感度。</li>
                    <li>边缘膨胀: 核大小/迭代次数，用于连接断线。</li>
                    <li>轮廓闭合: 核大小/迭代次数，用于填充内部。</li>
                    <li><i>(未来可能添加 GrabCut, Watershed 等更多算法)</i></li>
                 </ul>
            </li>
            <li>Home: 重置视图 (缩放/平移)</li>
            <li>Shift+拖动: 平移视图</li>
            <li>鼠标滚轮: 缩放视图</li>
            <li>Esc: 取消</li>
            <li>Ctrl+S / Enter: 保存并关闭</li>
        </ul>

        <p><b>命名模板占位符:</b> [索引], [X], [Y], [宽], [高], [面积], [长宽比], [标签], [备注]</p>
        <p><b>格式化示例:</b> [索引:03d] (补零到3位), [面积:.1f] (保留1位小数)</p>
        """
        
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("帮助")
        dialog.setMinimumSize(600, 550)
        layout = QtWidgets.QVBoxLayout(dialog)
        text_browser = QtWidgets.QTextBrowser()
        text_browser.setOpenExternalLinks(False)
        text_browser.setHtml(help_text)
        layout.addWidget(text_browser)
        close_btn = QtWidgets.QPushButton("关闭")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn, 0, Qt.AlignmentFlag.AlignCenter)
        dialog.exec()

    def _load_settings(self):
        """从QSettings加载应用设置。"""
        geometry = self.settings.value("geometry")
        if geometry:
            try:
                self.restoreGeometry(geometry)
            except Exception as e:
                print(f"Error restoring geometry: {e}")
            
        self.last_dir = self.settings.value("last_directory", ".")
        
        proc_params = {}
        for key in self.processor.get_params():
            value = self.settings.value(f"processor_{key}")
            if value is not None:
                default_value = getattr(self.processor, key, None)
                if isinstance(default_value, int):
                    try: proc_params[key] = int(value)
                    except ValueError: pass
                elif isinstance(default_value, float):
                    try: proc_params[key] = float(value)
                    except ValueError: pass
        self.processor.set_params(proc_params)

        self.sort_by = self.settings.value("sort_by", self.sort_by)
        self.sort_reverse = self.settings.value("sort_reverse", self.sort_reverse, type=bool)
        self.naming_template = self.settings.value("naming_template", self.naming_template)
        
        try:
            self.area_range = tuple(map(int, self.settings.value("area_range", self.area_range)))
        except: self.area_range = (0, 9999999)
        
        try:
            self.aspect_range = tuple(map(float, self.settings.value("aspect_range", self.aspect_range)))
        except: self.aspect_range = (0, 99.0)

    def _update_ui_from_settings(self):
        """根据加载的设置更新UI控件的初始状态。"""
        self.thresh_slider.setValue(self.processor.color_thresh)
        self.thresh_label.setText(f"色差阈值: {self.processor.color_thresh}")
        self.pad_spin.setValue(self.processor.pad)
        self.kernel_spin.setValue(self.processor.kernel_size)
        self.close_iter_spin.setValue(self.processor.close_iter)
        self.open_iter_spin.setValue(self.processor.open_iter)
        self.max_extract_spin.setValue(self.processor.max_extract)
        self.out_width_spin.setValue(self.processor.out_width)
        self.out_height_spin.setValue(self.processor.out_height)
        
        sort_map_inv = {v: k for k, v in {"idx": 0, "area": 1, "x": 2, "y": 3, "w": 4, "h": 5, "aspect_ratio": 6}.items()}
        if self.sort_by in sort_map_inv:
            self.sort_combo.setCurrentIndex(sort_map_inv[self.sort_by])
        else:
            self.sort_combo.setCurrentIndex(0)
            
        self.sort_order_combo.setCurrentIndex(0 if self.sort_reverse else 1)
        self.sort_order_combo.setEnabled(self.sort_by != 'idx')
        
        self.area_min_spin.setValue(self.area_range[0])
        self.area_max_spin.setValue(self.area_range[1])
        self.aspect_min_spin.setValue(self.aspect_range[0])
        self.aspect_max_spin.setValue(self.aspect_range[1])
        
        self.naming_input.setText(self.naming_template)
        self.on_naming_change()
        
    def closeEvent(self, event):
        """窗口关闭事件处理：保存应用设置。"""
        self.settings.setValue("geometry", self.saveGeometry())
        
        for key, value in self.processor.get_params().items():
            self.settings.setValue(f"processor_{key}", value)
            
        self.settings.setValue("sort_by", self.sort_by)
        self.settings.setValue("sort_reverse", self.sort_reverse)
        self.settings.setValue("area_range", list(self.area_range))
        self.settings.setValue("aspect_range", list(self.aspect_range))
        self.settings.setValue("naming_template", self.naming_template)
        self.settings.setValue("last_directory", self.last_dir)
        
        super().closeEvent(event)

    def keyPressEvent(self, event):
        """处理全局键盘快捷键。"""
        key = event.key()
        modifiers = event.modifiers()
        
        if modifiers == Qt.KeyboardModifier.ControlModifier:
            if key == Qt.Key.Key_O:
                self.load_image()
                event.accept()
            elif key == Qt.Key.Key_S:
                self.export_all()
                event.accept()
            elif key == Qt.Key.Key_P:
                self.show_animation_preview()
                event.accept()
            elif key == Qt.Key.Key_E:
                self.on_edit_mask()
                event.accept()
            else:
                super().keyPressEvent(event)
        else:
            super().keyPressEvent(event)

    def load_image(self):
        """槽函数：响应加载图片按钮点击，打开文件对话框并加载图像。"""
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, '选择图片', self.last_dir, 'Image Files (*.png *.webp *.jpg *.jpeg *.bmp *.gif)'
        )
        if not fname:
            return
        self.last_dir = os.path.dirname(fname)
        self.statusBar().showMessage(f"正在加载 {os.path.basename(fname)}...", 2000)
        QtWidgets.QApplication.processEvents()
        try:
            self.image = Image.open(fname).convert("RGBA")
            self.img_np = np.array(self.image)
            self.setWindowTitle(f'Sprite Mask 可视化操作台 - {os.path.basename(fname)}')
            self.refresh_mask_and_rois()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "加载失败", f"无法加载图片: {str(e)}")

    def refresh_mask_and_rois(self):
        """核心刷新逻辑：重新生成蒙版并提取、排序、筛选ROIs，然后更新UI。"""
        if self.img_np is None:
            return
        self.statusBar().showMessage("正在处理图像...", 0)
        QtWidgets.QApplication.processEvents()
        self.mask = self.processor.gen_mask(self.img_np)
        self._extract_and_update_rois()
        count = len(self.rois)
        self.statusBar().showMessage(f"已检测到 {count} 个区域", 3000)

    def refresh_rois_only(self):
        """仅重新提取、排序、筛选ROIs，不重新生成蒙版，然后更新UI。"""
        if self.img_np is None or self.mask is None:
            return
        self.statusBar().showMessage("正在提取区域...", 0)
        QtWidgets.QApplication.processEvents()
        self._extract_and_update_rois()
        count = len(self.rois)
        self.statusBar().showMessage(f"已检测到 {count} 个区域", 3000)

    def refresh_sort_filter(self):
        """仅重新应用排序和筛选，不重新提取ROI，然后更新UI。"""
        if not hasattr(self, '_all_rois') or not self._all_rois:
            return
        self.statusBar().showMessage("正在排序和筛选...", 0)
        QtWidgets.QApplication.processEvents()
        filtered_rois = filter_rois(self._all_rois, area_range=self.area_range, aspect_range=self.aspect_range)
        sorted_rois = sort_rois(filtered_rois, by=self.sort_by, reverse=self.sort_reverse)
        self.rois = sorted_rois
        self.thumb_list.set_thumbs([r.img for r in self.rois])
        if self.rois:
            self.current_idx = 0
            self.thumb_list.set_current(0)
            self.update_preview()
        else:
            self.current_idx = -1
            self.img_label.clear()
            self.mask_label.clear()
        self.on_naming_change()
        count = len(self.rois)
        self.statusBar().showMessage(f"显示 {count} 个已筛选区域", 3000)

    def _extract_and_update_rois(self):
        """提取ROIs，应用排序/筛选，并更新UI（缩略图列表和预览）。"""
        extracted_rois = self.processor.extract_rois(self.img_np, self.mask)
        self._all_rois = extracted_rois
        filtered_rois = filter_rois(extracted_rois, area_range=self.area_range, aspect_range=self.aspect_range)
        sorted_rois = sort_rois(filtered_rois, by=self.sort_by, reverse=self.sort_reverse)
        self.rois = sorted_rois
        self.thumb_list.set_thumbs([r.img for r in self.rois])
        if self.rois:
            self.current_idx = 0
            self.thumb_list.set_current(0)
            self.update_preview()
        else:
            self.current_idx = -1
            self.img_label.clear()
            self.mask_label.clear()
        self.on_naming_change()

    def insert_placeholder(self, placeholder):
        """将占位符插入命名模板输入框的当前光标位置。"""
        self.naming_input.insert(placeholder)

    def apply_preset_template(self, template):
        """将预设模板应用到命名模板输入框。"""
        self.naming_input.setText(template)
