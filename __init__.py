# BSAI_ComfyUI_Nodes
# Consolidated BSAI custom nodes for ComfyUI
# Duplicates removed, only unique functional nodes retained.

import importlib
import logging
import os

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

# =====================================================================
# 前端 JS 注入：用 <script> 标签直接注入到 HTML（和 BSAI_Premiere_Pro 同款方案）
# 双保险：中间件 + on_response_prepare 信号
# =====================================================================

_FOLDER_NAME = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
_VLP_JS = "bsai_vlp_main.js"
_VLP_JS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web", _VLP_JS)

import re
import time

_VLP_TAG_RE = re.compile(
    r'<script[^>]*src="/extensions/' + re.escape(_FOLDER_NAME) + r'/' + re.escape(_VLP_JS) + r'[^"]*"[^>]*></script>'
)


def _vlp_get_script_tag():
    """生成带版本号的 script 标签，确保浏览器总是加载最新版。"""
    if os.path.exists(_VLP_JS_PATH):
        mtime = os.path.getmtime(_VLP_JS_PATH)
        size = os.path.getsize(_VLP_JS_PATH)
        v = f"{mtime:.6f}_{size}"
    else:
        v = str(time.time())
    return f'<script src="/extensions/{_FOLDER_NAME}/{_VLP_JS}?v={v}"></script>'


def _vlp_inject_script_into_html(html):
    """将 script 标签注入到 HTML 的 </head> 前。"""
    tag = _vlp_get_script_tag()
    if tag in html:
        return None  # 已经是最新版本
    # 移除旧版本的标签
    html = _VLP_TAG_RE.sub('', html)
    if "</head>" in html:
        result = html.replace("</head>", tag + "</head>", 1)
    elif "</body>" in html:
        result = html.replace("</body>", tag + "</body>", 1)
    else:
        result = html + tag
    # 完整性校验：确保注入后 HTML 仍然完整
    if "</html>" not in result:
        print(f"[BSAI VLP] Injection sanity check failed, aborting injection")
        return None
    return result


# 机制 1：aiohttp 中间件
try:
    from aiohttp import web
    from server import PromptServer

    @web.middleware
    async def _bsai_vlp_middleware(request, handler):
        if request.path == "/":
            try:
                web_root = getattr(PromptServer.instance, "web_root", None)
                if not web_root:
                    import folder_paths
                    web_root = folder_paths.base_path
                html_path = os.path.join(web_root, "index.html")
                with open(html_path, "r", encoding="utf-8") as f:
                    html = f.read()
                modified = _vlp_inject_script_into_html(html)
                if modified is not None:
                    return web.Response(
                        text=modified,
                        content_type="text/html",
                        headers={
                            "Cache-Control": "no-store, must-revalidate",
                            "Pragma": "no-cache",
                            "Expires": "0",
                        },
                    )
            except Exception as e:
                print(f"[BSAI VLP] Middleware injection failed: {e}")
        return await handler(request)

    try:
        PromptServer.instance.app._middlewares.append(_bsai_vlp_middleware)
        print("[BSAI VLP] Middleware injection installed (Video Loader Plus)")
    except Exception as e:
        print(f"[BSAI VLP] Middleware append failed (app may be frozen): {e}")
except Exception as e:
    print(f"[BSAI VLP] Middleware setup failed: {e}")


# 机制 2：on_response_prepare 信号（app 冻结后也能用）
try:
    from server import PromptServer

    async def _bsai_vlp_on_response(request, response):
        path = getattr(request, "path", "")
        # 给 JS 文件加 no-cache 头
        if f"/extensions/{_FOLDER_NAME}/" in path and path.endswith(".js"):
            response.headers["Cache-Control"] = "no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
            return
        if path != "/":
            return
        content_type = getattr(response, "content_type", "") or ""
        if "text/html" not in content_type:
            return
        try:
            # 尝试获取 HTML 内容（兼容普通 Response 和 FileResponse）
            html = None
            body = response.body
            if isinstance(body, bytes):
                html = body.decode("utf-8", errors="ignore")
            elif isinstance(body, str):
                html = body
            else:
                # FileResponse 或其他流式响应：从文件读取
                file_path = getattr(response, "_path", None) or getattr(response, "path", None)
                if file_path and os.path.isfile(str(file_path)):
                    with open(str(file_path), "r", encoding="utf-8") as f:
                        html = f.read()

            if html is None:
                return

            modified = _vlp_inject_script_into_html(html)
            if modified is not None:
                modified_bytes = modified.encode("utf-8")
                response.body = modified_bytes
                # 确保 Content-Length 正确
                response.headers["Content-Length"] = str(len(modified_bytes))
                response.headers["Cache-Control"] = "no-store, must-revalidate"
                response.headers["Pragma"] = "no-cache"
        except Exception as e:
            print(f"[BSAI VLP] Response injection failed: {e}")

    PromptServer.instance.app.on_response_prepare.append(_bsai_vlp_on_response)
    print("[BSAI VLP] on_response_prepare injection installed (Video Loader Plus)")
except Exception as e:
    print(f"[BSAI VLP] on_response_prepare setup failed: {e}")
