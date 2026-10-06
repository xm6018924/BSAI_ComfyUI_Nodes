# BSAI_ComfyUI_Nodes
# Consolidated BSAI custom nodes for ComfyUI
# Duplicates removed, only unique functional nodes retained.

import importlib
import logging

logger = logging.getLogger(__name__)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

modules = [
    "MultiImageReverse",
    "AudioCropProcessUTK",
    "CompressImages",
    "BSAI_LongTextToList",
    "BSAI_ImageSequenceToVideo",
    "BSAI_VideoToImages",
    "BSAI_QwenNodes",
    "BSAI_AnyToList",
    "BSAI_MergeVideoAudioToImages",
    "BSAI_AudioDurationToFrames",
    "BSAI_PIPMultiLayer",
    "BSAI_MergeImages",
    "BSAI_Krea2Nodes",
    "BSAI_ComfyUI_lingbot_video",
    "BSAI_DrawTextOverlay",
]

for mod_name in modules:
    try:
        mod = importlib.import_module(f".{mod_name}", package=__name__)
        if hasattr(mod, 'NODE_CLASS_MAPPINGS'):
            NODE_CLASS_MAPPINGS.update(mod.NODE_CLASS_MAPPINGS)
        if hasattr(mod, 'NODE_DISPLAY_NAME_MAPPINGS'):
            NODE_DISPLAY_NAME_MAPPINGS.update(mod.NODE_DISPLAY_NAME_MAPPINGS)
        logger.info(f"Loaded {mod_name} nodes")
    except Exception as e:
        logger.warning(f"Failed to load {mod_name}: {e}")

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']

WEB_DIRECTORY = "./web"


# ==== BSAI VLP Sync: HTML 帧同步脚本注入（原 prestartup_script 迁移至此）====
# 原实现放在 prestartup_script.py：prestartup 阶段 ComfyUI server 尚未初始化，
# 只能 run_until_complete 循环重试 20 次（启动白等约 24 秒），且 from server import PromptServer
# 会提前导入 torch，触发 "Torch already imported" 警告。
# 现改为节点加载后由守护线程延迟安装：server 就绪后一次成功，不阻塞启动、不提前导入 torch，功能完全保留。
def _install_vlp_sync():
    import os
    try:
        from aiohttp import web
        from server import PromptServer
    except Exception:
        return
    server = None
    try:
        server = PromptServer.instance
    except Exception:
        server = None
    if not server or not hasattr(server, "app"):
        return
    web_dir = os.path.join(os.path.dirname(__file__), "web")
    js_file = os.path.join(web_dir, "bsai_vlp_sync.js")
    js_content = ""
    try:
        with open(js_file, "r", encoding="utf-8") as f:
            js_content = f.read()
    except Exception:
        return
    if not js_content.strip():
        return
    inline_script = "<script>\n// BSAI VLP Sync - injected by BSAI_ComfyUI_Nodes\n" + js_content + "\n</script>"

    @web.middleware
    async def bsai_vlp_middleware(request, handler):
        response = await handler(request)
        try:
            if response is None:
                return response
            ct = response.headers.get("Content-Type", "")
            if "text/html" not in ct:
                return response
            body = response.body
            if not body:
                return response
            try:
                html = body.decode("utf-8")
            except Exception:
                return response
            if "</head>" in html and "bsai_vlp_sync" not in html:
                html = html.replace("</head>", inline_script + "</head>", 1)
                response.body = html.encode("utf-8")
                if "Content-Length" in response.headers:
                    response.headers["Content-Length"] = str(len(response.body))
            return response
        except Exception:
            return response
    try:
        server.app.middlewares.append(bsai_vlp_middleware)
        print("[BSAI VLP Sync] Middleware injection installed (from __init__.py)")
        return True
    except Exception as e:
        print(f"[BSAI VLP Sync] Middleware install failed: {e}")
        return False


def _vlp_sync_installer():
    import time
    for _ in range(15):
        try:
            from server import PromptServer
        except Exception:
            time.sleep(2)
            continue
        if getattr(PromptServer, "instance", None) is not None:
            if _install_vlp_sync():
                break
        time.sleep(2)


import threading as _threading
_threading.Thread(target=_vlp_sync_installer, daemon=True, name="BSAI-VLP-Sync").start()
