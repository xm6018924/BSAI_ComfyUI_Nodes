"""
BSAI Video Loader Plus - Frame Sync Injection
在页面加载时把 JS 同步代码注入到 HTML 中。
"""
import os
from aiohttp import web
from server import PromptServer

WEB_DIR = os.path.join(os.path.dirname(__file__), "web")
JS_FILE = os.path.join(WEB_DIR, "bsai_vlp_sync.js")

def install_middleware():
    server = PromptServer.instance
    if not server or not hasattr(server, "app"):
        return False

    js_content = ""
    try:
        with open(JS_FILE, "r", encoding="utf-8") as f:
            js_content = f.read()
    except Exception:
        return False

    if not js_content.strip():
        return False

    # 内联脚本包裹
    inline_script = f"""
<script>
// BSAI VLP Sync - injected by prestartup
{js_content}
</script>
""".strip()

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
        print("[BSAI VLP Sync] Middleware injection installed")
        return True
    except Exception as e:
        print(f"[BSAI VLP Sync] Middleware install failed: {e}")
        return False

# 尝试多次安装，确保服务器就绪
import asyncio

_installed = False

async def _try_install():
    global _installed
    for _ in range(20):
        if _installed:
            break
        try:
            if install_middleware():
                _installed = True
                print("[BSAI VLP Sync] ✅ Frame Sync injection ready!")
                break
        except Exception:
            pass
        await asyncio.sleep(1)

if not _installed:
    loop = None
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if loop and loop.is_running():
        asyncio.ensure_future(_try_install())
    else:
        loop.run_until_complete(_try_install())
