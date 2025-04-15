# 图标目录

此目录用于存放应用程序的图标文件。

## 所需图标

请将以下名称的PNG图标文件放入此目录：

- open.png - 打开文件图标
- export.png - 导出图标
- preset.png - 预设管理图标
- save_preset.png - 保存预设图标
- load_preset.png - 加载预设图标
- play.png - 播放图标
- pause.png - 暂停图标
- prev.png - 上一帧图标
- next.png - 下一帧图标
- help.png - 帮助图标
- edit.png - 编辑图标
- brush.png - 画笔工具图标
- eraser.png - 橡皮工具图标
- undo.png - 撤销图标
- redo.png - 重做图标
- reset.png - 重置图标
- auto_fix.png - 自动修复图标
- save.png - 保存图标
- cancel.png - 取消图标
- batch_export.png - 批量导出图标
- batch_tag.png - 批量标签图标
- batch_note.png - 批量备注图标
- batch_import.png - 批量导入图标

## 如何编译资源

放置图标后，请运行 `tools/build_resources.py` 脚本来编译资源文件。

## 替代方案

如果无法使用PyQt的资源系统，可以修改代码以直接从文件系统加载图标：

```python
def get_icon(name):
    path = os.path.join(os.path.dirname(__file__), f"icons/{name}.png")
    if os.path.exists(path):
        return QIcon(path)
    return QIcon()
```

然后将代码中的 `QIcon(":/icons/name.png")` 替换为 `get_icon("name")`。 