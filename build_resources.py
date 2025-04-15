"""
---------------------------------------------------------------
File name:                  build_resources.py
Author:                     Ignorant-lu
Date created:               2025/04/15
Description:                生成PyQt资源文件的脚本，自动编译QRC为Python模块。
----------------------------------------------------------------

Changed history:            
                            2025/04/15: 初始创建;
----
"""
import os
import sys
import subprocess

def main():
    """主入口，检查依赖并编译资源文件

    检查PyQt6和icons目录，自动调用rcc工具生成resources.py。
    """
    # 检查是否安装了PyQt6
    try:
        import PyQt6
    except ImportError:
        print("错误: 请先安装PyQt6")
        return

    # 检查icons目录是否存在，如果不存在则创建
    if not os.path.exists("icons"):
        os.makedirs("icons")
        print("创建了 icons 目录")
        print("请将所需的图标放入 icons 目录，然后重新运行此脚本")
        return

    # 检查是否有图标文件
    icon_files = [f for f in os.listdir("icons") if f.endswith(".png")]
    if not icon_files:
        print("错误: icons 目录中没有找到PNG图标文件")
        print("请将图标文件放入 icons 目录，然后重新运行此脚本")
        return

    # 检查资源文件是否存在
    if not os.path.exists("resources.qrc"):
        print("错误: 没有找到 resources.qrc 文件")
        return

    # 使用PyQt的资源编译器编译资源文件
    try:
        # 获取PyQt6安装路径
        pyqt_path = os.path.dirname(PyQt6.__file__)
        
        # 在Windows上，我们使用pyrcc6.exe
        if sys.platform == "win32":
            rcc_path = os.path.join(pyqt_path, "rcc.exe")
            if not os.path.exists(rcc_path):
                rcc_path = os.path.join(pyqt_path, "Qt6", "bin", "rcc.exe")
        else:
            # 在Unix/Mac上，我们尝试使用命令行工具pyrcc6
            rcc_path = "pyrcc6"
        
        # 执行编译命令
        cmd = [rcc_path, "resources.qrc", "-o", "resources.py"]
        subprocess.run(cmd, check=True)
        print("成功编译资源文件: resources.py")
    except Exception as e:
        print(f"错误: 无法编译资源文件: {e}")
        print("提示: 请确保PyQt6已正确安装，并且rcc工具可用")
        print("替代方案: 如果无法编译资源，请创建一个空的resources.py文件")
        
        # 创建一个空的资源模块作为后备
        with open("resources.py", "w") as f:
            f.write("# 空的资源模块\n")
        print("已创建空的resources.py")

if __name__ == "__main__":
    # 保存当前工作目录
    original_cwd = os.getcwd()
    
    # 切换到脚本所在目录
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # 运行主函数
    main()
    
    # 恢复原来的工作目录
    os.chdir(original_cwd) 