/**
 * BSAI Video Loader Plus - 帧选择器扩展
 * 
 * 在 BSAI Video Loader Plus 节点中注入视频播放器，实现精准帧同步。
 * 参考 BSAI Video Merge 的实现模式。
 */

import { app } from "../../../scripts/app.js";

const STYLE_ID = "bsai-vlp-player-css";
if (!document.getElementById(STYLE_ID)) {
    const st = document.createElement("style");
    st.id = STYLE_ID;
    st.textContent = `
.bsai-vlp-player {
    padding: 8px;
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 6px;
    margin-top: 4px;
    color: #e2e8f0;
    font-family: system-ui, sans-serif;
    user-select: none;
}
.bsai-vlp-player-title {
    color: #38bdf8;
    font-size: 11px;
    font-weight: bold;
    margin-bottom: 6px;
}
.bsai-vlp-player video {
    width: 100%;
    background: #000;
    border-radius: 4px;
    max-height: 220px;
    object-fit: contain;
    display: block;
    margin-bottom: 4px;
}
.bsai-vlp-player-info {
    color: #64748b;
    font-size: 10px;
    display: flex;
    justify-content: space-between;
    font-family: monospace;
    margin-bottom: 4px;
}
.bsai-vlp-player-btn {
    width: 100%;
    padding: 6px;
    background: #0ea5e9;
    color: #fff;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 11px;
    font-weight: bold;
}
.bsai-vlp-player-btn:hover {
    background: #0284c7;
}
.bsai-vlp-player-hint {
    color: #475569;
    font-size: 9px;
    margin-top: 4px;
    text-align: center;
}
.bsai-vlp-debug {
    position: fixed;
    top: 8px;
    left: 8px;
    width: 36px;
    height: 24px;
    line-height: 22px;
    text-align: center;
    font-size: 11px;
    font-weight: bold;
    background: #10b981;
    color: #fff;
    border-radius: 6px;
    z-index: 99999;
    cursor: pointer;
    user-select: none;
    box-shadow: 0 2px 8px rgba(0,0,0,.3);
}
`;
    document.head.appendChild(st);
}

// 调试标记
const debugBadge = document.createElement("div");
debugBadge.className = "bsai-vlp-debug";
debugBadge.textContent = "VLP";
debugBadge.title = "BSAI VLP Player - 点击查看调试";
debugBadge.onclick = function() {
    const graph = app.canvas?.graph || app.graph;
    let info = "BSAI VLP 调试\n==========\n";

    // 直接扫描 DOM 节点
    const domNodes = document.querySelectorAll(".litegraph-node");
    info += "DOM 节点数: " + domNodes.length + "\n\n";

    for (let i = 0; i < domNodes.length; i++) {
        const d = domNodes[i];
        const id = d.getAttribute("data-id");
        const title = d.querySelector(".title")?.textContent || "(无标题)";
        const hasPlayer = d.querySelector(".bsai-vlp-player") ? "✅" : "❌";
        info += "#" + id + " " + title.substring(0, 30) + " 播放器:" + hasPlayer + "\n";

        // 扫描所有子元素，找出带 video/canvas 的
        const allVids = d.querySelectorAll("video");
        const allCavs = d.querySelectorAll("canvas");
        info += "  video: " + allVids.length + "  canvas: " + allCavs.length + "\n";

        // 打印 .widgets 下的完整结构
        const widgetsEl = d.querySelector(".widgets");
        if (widgetsEl) {
            info += "  --- .widgets 子元素 ---\n";
            for (let w = 0; w < widgetsEl.children.length; w++) {
                const child = widgetsEl.children[w];
                const rect = child.getBoundingClientRect();
                const tag = child.tagName;
                const cls = (child.className || "").substring(0, 50);
                const hasVid = child.querySelector("video, canvas") ? " 🎥" : "";
                const isOur = child.classList?.contains("bsai-vlp-player") ? " [我方]" : "";
                info += "  [" + w + "] " + tag + "." + cls + " [" + Math.round(rect.width) + "x" + Math.round(rect.height) + "]" + hasVid + isOur + "\n";

                // 如果有视频，再深入一层
                if (hasVid && !isOur) {
                    const innerKids = child.children;
                    for (let k = 0; k < innerKids.length && k < 10; k++) {
                        const ik = innerKids[k];
                        const irect = ik.getBoundingClientRect();
                        const itag = ik.tagName;
                        const icls = (ik.className || "").substring(0, 40);
                        info += "      " + itag + "." + icls + " [" + Math.round(irect.width) + "x" + Math.round(irect.height) + "]\n";
                    }
                }
            }
        }
        info += "\n";
    }

    // Graph 信息
    if (graph) {
        const nodes = graph._nodes || graph.nodes || [];
        info += "\nGraph 节点: " + nodes.length + "\n";
        for (let i = 0; i < nodes.length && i < 5; i++) {
            const n = nodes[i];
            info += "  #" + n.id + " " + n.type + " (widgets:" + (n.widgets?.length || 0) + ")\n";
            if (n.widgets) {
                for (let w = 0; w < n.widgets.length; w++) {
                    const ww = n.widgets[w];
                    const hasEl = ww.element ? " [element]" : (ww.inputEl ? " [inputEl]" : "");
                    info += "    [" + w + "] " + ww.name + " (" + ww.type + ")" + hasEl + "\n";
                }
            }
            if (n.__bsai_vlp_player) info += "    ✅ 已注入播放器\n";
        }
    }

    alert(info);
    console.log("[BSAI VLP] Debug:\n" + info);
};
document.body.appendChild(debugBadge);

// 工具函数
function getWidgetByName(node, name) {
    if (!node || !node.widgets) return null;
    for (let i = 0; i < node.widgets.length; i++) {
        if (node.widgets[i].name === name) return node.widgets[i];
    }
    return null;
}

function getVideoUrl(node) {
    const fileW = getWidgetByName(node, "file");
    if (!fileW || !fileW.value) return null;

    let filename = fileW.value;

    // 处理对象格式
    if (typeof filename === "object" && filename.filename) {
        const f = filename;
        let url = "/view?filename=" + encodeURIComponent(f.filename);
        if (f.subfolder) url += "&subfolder=" + encodeURIComponent(f.subfolder);
        url += "&type=" + (f.type || "input");
        return url;
    }

    // 字符串格式
    let subfolder = "";
    try {
        if (fileW.options && fileW.options.subfolder) {
            subfolder = fileW.options.subfolder;
        }
    } catch(e) {}

    let url = "/view?filename=" + encodeURIComponent(filename) + "&type=input";
    if (subfolder) url += "&subfolder=" + encodeURIComponent(subfolder);

    return url;
}

function setFrameIndex(node, frameNum) {
    const w = getWidgetByName(node, "frame_index");
    if (!w) return false;

    let maxVal = 99999;
    try {
        if (w.options && w.options.max !== undefined) maxVal = w.options.max;
        else if (w.max !== undefined) maxVal = w.max;
    } catch(e) {}

    frameNum = Math.max(0, Math.min(frameNum, maxVal));
    w.value = frameNum;

    if (w.callback) {
        try { w.callback(frameNum); } catch(e) {}
    }

    if (w.inputEl) {
        w.inputEl.value = frameNum;
        try {
            const ev = new Event("input", { bubbles: true });
            w.inputEl.dispatchEvent(ev);
            const ev2 = new Event("change", { bubbles: true });
            w.inputEl.dispatchEvent(ev2);
        } catch(e) {}
    }

    if (node.graph && node.graph.setDirtyCanvas) {
        node.graph.setDirtyCanvas(true, true);
    }

    return true;
}

function getFps(node) {
    const w = getWidgetByName(node, "frame_rate");
    if (w && w.value) {
        const f = parseFloat(w.value);
        if (f > 0) return f;
    }
    return 30.0;
}

// 创建播放器
function createPlayer(node) {
    const container = document.createElement("div");
    container.className = "bsai-vlp-player";

    const title = document.createElement("div");
    title.className = "bsai-vlp-player-title";
    title.textContent = "🎬 BSAI 帧选择器";
    container.appendChild(title);

    const video = document.createElement("video");
    video.controls = true;
    video.playsInline = true;
    video.preload = "auto";
    container.appendChild(video);

    const info = document.createElement("div");
    info.className = "bsai-vlp-player-info";
    info.innerHTML = "<span>⏱ 0.00s</span><span>🎞 帧 0</span><span class='bsai-fps-tag'>检测中...</span>";
    container.appendChild(info);

    const syncBtn = document.createElement("button");
    syncBtn.className = "bsai-vlp-player-btn";
    syncBtn.textContent = "📸 同步当前帧到 Frame Index";
    container.appendChild(syncBtn);

    const hint = document.createElement("div");
    hint.className = "bsai-vlp-player-hint";
    hint.textContent = "💡 暂停视频自动同步帧号";
    container.appendChild(hint);

    // ---- 真实 FPS 探测（用 requestVideoFrameCallback，100% 准确） ----
    let realFps = null;       // 探测到的真实 FPS
    let fpsSamples = [];      // FPS 样本（取平均值更稳定）
    let lastFrameCount = 0;
    let lastFrameTime = 0;

    function updateFpsTag() {
        const tag = info.querySelector(".bsai-fps-tag");
        if (!tag) return;
        if (realFps !== null) {
            tag.textContent = "真实 " + realFps.toFixed(2) + "fps";
            tag.style.color = "#4ade80";
        } else {
            tag.textContent = "FPS检测中...";
            tag.style.color = "#fbbf24";
        }
    }

    function probeFpsWithRVFC() {
        if (!video.requestVideoFrameCallback) {
            // 浏览器不支持 RVFC，兜底用 videoDuration + 估算
            realFps = getFps(node);
            updateFpsTag();
            return;
        }

        function onFrame(now, metadata) {
            if (metadata.presentedFrames > 0 && metadata.mediaTime > 0.1) {
                const fps = metadata.presentedFrames / metadata.mediaTime;
                if (fps > 10 && fps < 240) {
                    fpsSamples.push(fps);
                    // 取最近 10 个样本的平均值
                    if (fpsSamples.length > 10) fpsSamples.shift();
                    if (fpsSamples.length >= 3) {
                        const avg = fpsSamples.reduce(function(a, b) { return a + b; }, 0) / fpsSamples.length;
                        realFps = Math.round(avg * 100) / 100;
                        updateFpsTag();
                    }
                }
            }
            video.requestVideoFrameCallback(onFrame);
        }
        video.requestVideoFrameCallback(onFrame);
    }

    // ---- 计算当前帧号（优先用真实 FPS，兜底用设置的 frame_rate） ----
    function getCurrentFrame() {
        const ct = video.currentTime || 0;
        const fps = realFps !== null ? realFps : getFps(node);
        return Math.floor(ct * fps);
    }

    function getCurrentFps() {
        return realFps !== null ? realFps : getFps(node);
    }

    // 视频事件
    let lastSync = 0;

    function syncFrame() {
        const now = Date.now();
        if (now - lastSync < 100) return;
        lastSync = now;

        const ct = video.currentTime || 0;
        const frame = getCurrentFrame();
        const fps = getCurrentFps();

        info.innerHTML = "<span>⏱ " + ct.toFixed(2) + "s</span>" +
                       "<span>🎞 帧 " + frame + "</span>" +
                       "<span class='bsai-fps-tag'>" + (realFps !== null ? "真实 " + realFps.toFixed(2) + "fps" : "FPS检测中...") + "</span>";

        if (video.paused) {
            setFrameIndex(node, frame);
        }
    }

    video.addEventListener("timeupdate", function() {
        syncFrame();
    });

    video.addEventListener("loadedmetadata", function() {
        // 元数据加载后，开始 FPS 探测
        probeFpsWithRVFC();
        // 也尝试播放一帧来加速探测
        if (video.paused) {
            video.currentTime = 0.1;
        }
    });

    video.addEventListener("pause", function() {
        setTimeout(syncFrame, 80);
        syncBtn.textContent = "✅ 已同步";
        setTimeout(function() { syncBtn.textContent = "📸 同步当前帧到 Frame Index"; }, 1200);
    });

    video.addEventListener("seeked", function() {
        if (video.paused) setTimeout(syncFrame, 80);
    });

    syncBtn.addEventListener("click", function(e) {
        e.stopPropagation();
        const frame = getCurrentFrame();
        setFrameIndex(node, frame);
        syncBtn.textContent = "✅ 已同步到帧 " + frame;
        setTimeout(function() { syncBtn.textContent = "📸 同步当前帧到 Frame Index"; }, 1500);
    });

    video.addEventListener("error", function() {
        hint.style.color = "#ef4444";
        hint.textContent = "⚠️ 视频加载失败";
    });

    video.addEventListener("loadedmetadata", function() {
        hint.style.color = "#475569";
        hint.textContent = "💡 暂停视频自动同步帧号";
    });

    // 防止拖动节点
    container.addEventListener("mousedown", function(e) {
        e.stopPropagation();
    });

    const player = {
        container: container,
        video: video,
        info: info,
        hint: hint,
        syncBtn: syncBtn,
        currentSrc: "",
    };

    node.__bsai_vlp_player = player;

    // 监听 file 变化
    const fileWidget = getWidgetByName(node, "file");
    if (fileWidget) {
        const origCallback = fileWidget.callback;
        fileWidget.callback = function() {
            setTimeout(function() { updatePlayerVideo(node); }, 100);
            if (origCallback) {
                try { return origCallback.apply(this, arguments); } catch(e) {}
            }
        };
    }

    return player;
}

function updatePlayerVideo(node) {
    const player = node.__bsai_vlp_player;
    if (!player) return;

    const url = getVideoUrl(node);
    if (!url) {
        player.hint.textContent = "请先选择视频文件";
        player.hint.style.color = "#64748b";
        return;
    }

    if (player.currentSrc !== url) {
        player.currentSrc = url;
        player.video.src = url;
        player.video.load();
        player.hint.textContent = "加载中...";
        player.hint.style.color = "#64748b";
    }
}

// 注入播放器到节点（优先用 addDOMWidget，失败则用 DOM 注入兜底）
function injectPlayerDOM(node) {
    if (node.__bsai_vlp_injected) return;
    node.__bsai_vlp_injected = true;

    console.log("[BSAI VLP] 开始注入节点 #" + node.id + " 类型: " + node.type);

    // 先创建播放器 DOM
    const player = createPlayer(node);

    // 方案1: 尝试 addDOMWidget
    if (typeof node.addDOMWidget === "function") {
        try {
            const widget = node.addDOMWidget(
                "bsai_frame_picker",
                "bsai_video_player",
                player.container,
                {
                    getHeight: function() { return 310; },
                    hideOnZoom: false,
                }
            );
            widget.serializeValue = function() { return undefined; };

            node.__bsai_vlp_widget = widget;
            console.log("[BSAI VLP] addDOMWidget 成功 #" + node.id);

            // 隐藏内置视频预览 + 持续监控
            hideBuiltinVideoPreview(node);
            startHideMonitor(node);

            // 加载视频
            setTimeout(() => updatePlayerVideo(node), 100);

            // 重绘
            if (node.onResize) {
                try { node.onResize(node.size); } catch(e) {}
            }
            if (node.graph && node.graph.setDirtyCanvas) {
                node.graph.setDirtyCanvas(true, true);
            }
            return;
        } catch(e) {
            console.error("[BSAI VLP] addDOMWidget 失败:", e);
        }
    }

    // 方案2: DOM 注入兜底
    console.log("[BSAI VLP] 尝试 DOM 注入兜底 #" + node.id);
    injectPlayerDOMFallback(node, player, 0);
}

// 隐藏节点内的内置视频预览
function hideBuiltinVideoPreview(node, nodeEl) {
    let hiddenCount = 0;

    if (!node.widgets) return;

    // 方案0: 从 widgets 数组中移除 video-preview widget（最彻底，直接消除占用空间）
    let removedFromArray = false;
    for (let i = node.widgets.length - 1; i >= 0; i--) {
        const w = node.widgets[i];
        if (w.name === "video-preview" ||
            (w.type === "video" && w.name !== "bsai_frame_picker")) {

            // 隐藏 DOM 元素
            if (w.element) {
                w.element.style.display = "none";
                w.element.style.height = "0px";
                w.element.style.margin = "0";
                w.element.style.padding = "0";
                w.element.style.border = "none";
                w.element.style.overflow = "hidden";
            }

            // 从数组移除（消除空间占用）
            node.widgets.splice(i, 1);
            removedFromArray = true;
            hiddenCount++;
            console.log("[BSAI VLP] 从 widgets 数组移除:", w.name, "(类型:", w.type + ")");
        }
    }

    if (removedFromArray) {
        // 强制节点重新计算尺寸
        setTimeout(function() {
            if (node.onResize) {
                try { node.onResize(node.size); } catch(e) {}
            }
            if (node.setSize) {
                try {
                    // 触发尺寸重新计算
                    const w = node.size[0];
                    node.setSize([w, 1]);
                    setTimeout(function() {
                        if (node.onResize) {
                            try { node.onResize(node.size); } catch(e) {}
                        }
                        if (node.graph && node.graph.setDirtyCanvas) {
                            node.graph.setDirtyCanvas(true, true);
                        }
                    }, 30);
                } catch(e) {}
            }
            if (node.graph && node.graph.setDirtyCanvas) {
                node.graph.setDirtyCanvas(true, true);
            }
        }, 50);
    }

    // 方案1: 隐藏 file widget 的预览部分
    const fileWidget = getWidgetByName(node, "file");
    if (fileWidget) {
        const widgetEl = fileWidget.element || fileWidget.inputEl?.parentElement;
        if (widgetEl) {
            const children = widgetEl.children || widgetEl.childNodes;
            for (let i = 0; i < children.length; i++) {
                const child = children[i];
                if (child.nodeType !== 1) continue;
                const tag = child.tagName?.toLowerCase() || "";
                const cls = (child.className || "").toString().toLowerCase();
                if (tag === "input" || tag === "select" || tag === "button") continue;
                if (cls.indexOf("input") !== -1 && cls.indexOf("preview") === -1) continue;

                const rect = child.getBoundingClientRect();
                if (rect.width > 80 && rect.height > 40) {
                    const hasVideo = child.querySelector("video, canvas");
                    const hasPreviewCls = cls.indexOf("preview") !== -1 || cls.indexOf("video") !== -1;
                    if ((hasVideo || hasPreviewCls) && !child.closest?.(".bsai-vlp-player")) {
                        child.style.display = "none";
                        child.style.height = "0";
                        child.style.margin = "0";
                        child.style.padding = "0";
                        hiddenCount++;
                        console.log("[BSAI VLP] 隐藏 file widget 预览:", tag + "." + cls.substring(0, 40));
                    }
                }
            }
        }
    }

    // 方案2: 全局搜索 video-preview 元素（Canvas 渲染模式下，widget 是浮在页面上的）
    const allPreviewEls = document.querySelectorAll(
        "[class*='video-preview'], [class*='videoPreview'], [class*='VideoPreview'], [class*='videopreview']"
    );
    for (const el of allPreviewEls) {
        if (el.closest?.(".bsai-vlp-player")) continue;
        const rect = el.getBoundingClientRect();
        if (rect.width > 100 && rect.height > 50) {
            el.style.display = "none";
            el.style.height = "0";
            el.style.margin = "0";
            el.style.padding = "0";
            hiddenCount++;
            console.log("[BSAI VLP] 全局隐藏预览元素:", el.className?.substring(0, 50), rect.width + "x" + rect.height);
        }
    }

    if (hiddenCount > 0) {
        console.log("[BSAI VLP] 共隐藏 " + hiddenCount + " 个内置预览元素");
    }
}

// 持续监控并隐藏 video-preview（防止异步创建后又显示）
function startHideMonitor(node) {
    if (node.__bsai_hide_monitor) return;
    node.__bsai_hide_monitor = true;

    // 定期隐藏
    let hideTries = 0;
    const hideTimer = setInterval(function() {
        hideTries++;
        hideBuiltinVideoPreview(node);
        // 30 次后降低频率
        if (hideTries > 30) {
            clearInterval(hideTimer);
            setInterval(function() { hideBuiltinVideoPreview(node); }, 3000);
        }
    }, 500);

    // 监听 video-preview widget 的 element 变化
    const vpWidget = getWidgetByName(node, "video-preview");
    if (vpWidget) {
        // 如果 element 还没创建，用 Object.defineProperty 监听
        if (!vpWidget.element) {
            let _element = vpWidget.element;
            Object.defineProperty(vpWidget, "element", {
                get: function() { return _element; },
                set: function(val) {
                    _element = val;
                    if (val) {
                        val.style.display = "none";
                        console.log("[BSAI VLP] 拦截 video-preview element 创建，立即隐藏");
                    }
                },
                configurable: true,
            });
        }
    }

    // MutationObserver 全局监控
    const observer = new MutationObserver(function(mutations) {
        for (let i = 0; i < mutations.length; i++) {
            const mut = mutations[i];
            for (let j = 0; j < mut.addedNodes.length; j++) {
                const node = mut.addedNodes[j];
                if (node.nodeType !== 1) continue;

                // 检查新增元素本身
                if (isVideoPreviewEl(node)) {
                    node.style.display = "none";
                    console.log("[BSAI VLP] MutationObserver 隐藏预览:", node.className?.substring(0, 40));
                }
                // 检查子元素
                const innerPreviews = node.querySelectorAll?.("[class*='video-preview'], [class*='VideoPreview'], [class*='videopreview']");
                if (innerPreviews) {
                    for (let k = 0; k < innerPreviews.length; k++) {
                        if (!innerPreviews[k].closest?.(".bsai-vlp-player")) {
                            innerPreviews[k].style.display = "none";
                        }
                    }
                }
            }
        }
    });

    function isVideoPreviewEl(el) {
        if (!el.className) return false;
        const cls = el.className.toString().toLowerCase();
        return (cls.indexOf("video-preview") !== -1 ||
                cls.indexOf("videopreview") !== -1) &&
               !cls.contains?.("bsai");
    }

    try {
        observer.observe(document.body, { childList: true, subtree: true });
    } catch(e) {}
}

function injectPlayerDOMFallback(node, player, tries) {
    if (tries > 20) {
        console.error("[BSAI VLP] DOM 注入失败，已达最大重试次数 #" + node.id);
        return;
    }

    const nodeEl = document.querySelector('.litegraph-node[data-id="' + node.id + '"]');
    if (!nodeEl) {
        setTimeout(() => injectPlayerDOMFallback(node, player, tries + 1), 200);
        return;
    }

    const widgetsEl = nodeEl.querySelector(".widgets");
    if (!widgetsEl) {
        setTimeout(() => injectPlayerDOMFallback(node, player, tries + 1), 200);
        return;
    }

    if (widgetsEl.querySelector(".bsai-vlp-player")) {
        updatePlayerVideo(node);
        return;
    }

    widgetsEl.appendChild(player.container);
    console.log("[BSAI VLP] DOM 注入成功 #" + node.id);

    // 隐藏内置视频预览 + 持续监控
    hideBuiltinVideoPreview(node, nodeEl);
    startHideMonitor(node);

    updatePlayerVideo(node);

    setTimeout(function() {
        if (node.onResize) {
            try { node.onResize(node.size); } catch(e) {}
        }
        if (node.graph && node.graph.setDirtyCanvas) {
            node.graph.setDirtyCanvas(true, true);
        }
    }, 50);
}

// 注册扩展
app.registerExtension({
    name: "BSAI.VideoLoaderPlus.FramePicker",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // 匹配 BSAI Video Loader Plus 节点
        const name = nodeData.name || "";
        if (name === "BSAI_VideoLoaderPlus" ||
            (name.indexOf("BSAI") !== -1 &&
             name.toLowerCase().indexOf("video") !== -1 &&
             name.toLowerCase().indexOf("loader") !== -1)) {

            console.log("[BSAI VLP] 注册节点扩展:", name);

            // 节点添加到画布时
            const onAdded = nodeType.prototype.onAdded;
            nodeType.prototype.onAdded = function() {
                const r = onAdded ? onAdded.apply(this, arguments) : undefined;
                const node = this;
                setTimeout(() => injectPlayerDOM(node), 200);
                setTimeout(() => injectPlayerDOM(node), 600);
                return r;
            };

            // 配置节点时（加载工作流）
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function() {
                const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
                const node = this;
                setTimeout(() => injectPlayerDOM(node), 300);
                setTimeout(() => injectPlayerDOM(node), 800);
                return r;
            };
        }
    },
});

console.log("[BSAI VLP] 扩展已注册 (ES module)");

// ===== 全局扫描兜底（防止 onAdded/onConfigure 钩子不生效） =====
function scanAndInjectAll() {
    const graph = app.canvas?.graph || app.graph;
    if (!graph) return;

    const nodes = graph._nodes || graph.nodes || [];
    for (let i = 0; i < nodes.length; i++) {
        const n = nodes[i];
        const type = n.type || "";
        if (type.indexOf("BSAI") !== -1 &&
            type.toLowerCase().indexOf("video") !== -1 &&
            type.toLowerCase().indexOf("loader") !== -1) {
            if (!n.__bsai_vlp_player) {
                console.log("[BSAI VLP] 全局扫描发现节点:", type, "#" + n.id);
                injectPlayerDOM(n);
            }
        }
    }
}

// 启动后持续扫描
let scanCount = 0;
const scanTimer = setInterval(function() {
    scanCount++;
    scanAndInjectAll();
    // 扫描 15 次后停止密集扫描，改为每 5 秒一次
    if (scanCount > 15) {
        clearInterval(scanTimer);
        setInterval(scanAndInjectAll, 5000);
    }
}, 1000);

console.log("[BSAI VLP] 全局扫描已启动");
