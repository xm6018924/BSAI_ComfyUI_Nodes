"""
BSAI_ComfyUI_Nodes - VLP Sync 注入（已迁移）

原实现：在 prestartup 阶段 `from server import PromptServer` 并 run_until_complete
循环重试 20 次安装 middleware，导致两个问题：
  1. prestartup 阶段 ComfyUI server 尚未初始化，启动白等约 24 秒；
  2. `from server import PromptServer` 会提前导入 torch，
     触发 main.py 的 "Torch already imported" 警告。

现迁移至 __init__.py（节点加载后由守护线程延迟安装，server 就绪后一次成功），
功能完全保留（HTML 帧同步脚本注入），本文件保留为空操作占位，不再导入任何模块。
"""
