/**
 * BSAI Video Loader Plus - Main Enhancement
 * 为 BSAI Video Loader Plus 节点添加帧选择功能。
 * 通过 <script> 标签加载，使用全局变量，兼容所有 ComfyUI 前端版本。
 */
if (window.__bsai_vlp_main_loaded) {
    // 防止重复加载
} else {
window.__bsai_vlp_main_loaded = true;
console.log("[BSAI VLP] Main script v1");

// ============================================================
// CSS 样式
// ============================================================
(function injectCSS() {
    if (document.getElementById("bsai-vlp-plus-css")) return;
    var st = document.createElement("style");
    st.id = "bsai-vlp-plus-css";
    st.textContent = [
        '.bsai-vlp-panel {',
        '    margin: 8px 12px 12px;',
        '    padding: 10px 12px;',
        '    background: linear-gradient(180deg, #1e2a3a 0%, #141e2c 100%);',
        '    border: 1px solid #2a3a4a;',
        '    border-radius: 8px;',
        '    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;',
        '    font-size: 12px;',
        '    color: #ccc;',
        '    user-select: none;',
        '}',
        '.bsai-vlp-header {',
        '    display: flex;',
        '    align-items: center;',
        '    gap: 6px;',
        '    margin-bottom: 8px;',
        '}',
        '.bsai-vlp-title {',
        '    color: #7ab8ff;',
        '    font-weight: bold;',
        '    font-size: 13px;',
        '}',
        '.bsai-vlp-badge {',
        '    margin-left: auto;',
        '    padding: 2px 10px;',
        '    border-radius: 10px;',
        '    font-size: 10px;',
        '    font-weight: bold;',
        '}',
        '.bsai-vlp-badge.full {',
        '    background: rgba(100,120,140,0.3);',
        '    color: #aabbcc;',
        '}',
        '.bsai-vlp-badge.single {',
        '    background: rgba(74,154,90,0.4);',
        '    color: #8be89b;',
        '}',
        '.bsai-vlp-row {',
        '    display: flex;',
        '    align-items: center;',
        '    gap: 8px;',
        '    margin-bottom: 8px;',
        '}',
        '.bsai-vlp-row:last-child { margin-bottom: 0; }',
        '.bsai-vlp-btn {',
        '    flex: 1;',
        '    background: linear-gradient(180deg, #4a9a5a 0%, #3a8a4a 100%);',
        '    color: #fff;',
        '    border: 1px solid #5aaa6a;',
        '    padding: 8px 12px;',
        '    border-radius: 5px;',
        '    cursor: pointer;',
        '    font-size: 13px;',
        '    font-weight: 600;',
        '    transition: all 0.15s;',
        '}',
        '.bsai-vlp-btn:hover {',
        '    background: linear-gradient(180deg, #5aaa6a 0%, #4a9a5a 100%);',
        '    border-color: #6aba7a;',
        '    transform: translateY(-1px);',
        '}',
        '.bsai-vlp-btn:active {',
        '    background: linear-gradient(180deg, #3a8a4a 0%, #2a7a3a 100%);',
        '    transform: translateY(0);',
        '}',
        '.bsai-vlp-btn.flash {',
        '    background: linear-gradient(180deg, #f0c040 0%, #d0a030 100%) !important;',
        '    border-color: #f0d060 !important;',
        '    color: #000 !important;',
        '}',
        '.bsai-vlp-small-btn {',
        '    background: #2a3a4a;',
        '    color: #aac8e8;',
        '    border: 1px solid #3a4a5a;',
        '    padding: 4px 10px;',
        '    border-radius: 4px;',
        '    cursor: pointer;',
        '    font-size: 12px;',
        '    transition: all 0.15s;',
        '    min-width: 30px;',
        '    text-align: center;',
        '}',
        '.bsai-vlp-small-btn:hover {',
        '    background: #3a4a5a;',
        '    border-color: #4a5a6a;',
        '}',
        '.bsai-vlp-small-btn:active { background: #1a2a3a; }',
        '.bsai-vlp-input {',
        '    background: #0f1a2a;',
        '    border: 1px solid #2a3a4a;',
        '    color: #eee;',
        '    padding: 4px 8px;',
        '    border-radius: 4px;',
        '    font-size: 12px;',
        '    width: 70px;',
        '    text-align: center;',
        '}',
        '.bsai-vlp-input:focus {',
        '    outline: none;',
        '    border-color: #4a90d9;',
        '    box-shadow: 0 0 0 2px rgba(74,144,217,0.25);',
        '}',
        '.bsai-vlp-info {',
        '    color: #8899aa;',
        '    font-size: 11px;',
        '    flex: 1;',
        '    text-align: right;',
        '}',
        '.bsai-vlp-hint {',
        '    color: #667788;',
        '    font-size: 11px;',
        '    margin-top: 8px;',
        '    text-align: center;',
        '    font-style: italic;',
        '}',
    ].join("\n");
    document.head.appendChild(st);
})();

// ============================================================
// 工具函数
// ============================================================

function getWidget(node, name) {
    if (!node || !node.widgets) return null;
    for (var i = 0; i < node.widgets.length; i++) {
        if (node.widgets[i].name === name) return node.widgets[i];
    }
    return null;
}

function setWidgetValue(node, name, value) {
    var w = getWidget(node, name);
    if (!w) return false;
    w.value = value;
    if (w.callback) {
        try { w.callback(value, w, node); } catch (e) {}
    }
    if (node.setDirtyCanvas) node.setDirtyCanvas(true, true);
    if (node.onWidgetChanged) {
        try { node.onWidgetChanged(name, value, w); } catch (e) {}
    }
    return true;
}

function findNodeElement(node) {
    if (!node || node.id == null) return null;
    // 兼容多种 id 属性
    var sel = '.litegraph-node[id="' + node.id + '"], ' +
              '.litegraph-node[data-id="' + node.id + '"]';
    var el = document.querySelector(sel);
    if (el) return el;
    // 退而求其次：通过 title 匹配
    var nodes = document.querySelectorAll(".litegraph-node");
    for (var i = 0; i < nodes.length; i++) {
        var titleEl = nodes[i].querySelector(".title");
        if (titleEl && titleEl.textContent &&
            titleEl.textContent.indexOf("Video Loader Plus") !== -1 &&
            node.title && titleEl.textContent.indexOf(node.title) !== -1) {
            return nodes[i];
        }
    }
    return null;
}

function findVideoInNode(nodeEl) {
    if (!nodeEl) return null;
    return nodeEl.querySelector("video");
}

function isVideoLoaderPlusNode(node) {
    if (!node) return false;
    var type = node.type || node.comfyClass || "";
    var title = node.title || "";
    return (type === "BSAI_VideoLoaderPlus" || type === "VideoLoaderPlus" ||
            title === "BSAI Video Loader Plus" || title === "Video Loader Plus");
}

// ============================================================
// 创建帧选择面板
// ============================================================

function createFramePanel(node) {
    var panel = document.createElement("div");
    panel.className = "bsai-vlp-panel";
    panel.setAttribute("data-bsvlp", "1");

    // 头部
    var header = document.createElement("div");
    header.className = "bsai-vlp-header";
    var title = document.createElement("span");
    title.className = "bsai-vlp-title";
    title.textContent = "🎬 帧选择工具";
    var badge = document.createElement("span");
    badge.className = "bsai-vlp-badge full";
    badge.textContent = "完整视频";
    header.appendChild(title);
    header.appendChild(badge);

    // 帧号控制行
    var row1 = document.createElement("div");
    row1.className = "bsai-vlp-row";

    var prevBtn = document.createElement("button");
    prevBtn.className = "bsai-vlp-small-btn";
    prevBtn.textContent = "◀";
    prevBtn.title = "上一帧";

    var frameInput = document.createElement("input");
    frameInput.type = "number";
    frameInput.className = "bsai-vlp-input";
    frameInput.min = "0";
    frameInput.value = "0";
    frameInput.title = "帧索引（从 0 开始），回车确认";

    var nextBtn = document.createElement("button");
    nextBtn.className = "bsai-vlp-small-btn";
    nextBtn.textContent = "▶";
    nextBtn.title = "下一帧";

    var frameInfo = document.createElement("span");
    frameInfo.className = "bsai-vlp-info";
    frameInfo.textContent = "帧";

    row1.appendChild(prevBtn);
    row1.appendChild(frameInput);
    row1.appendChild(nextBtn);
    row1.appendChild(frameInfo);

    // 主按钮行
    var row2 = document.createElement("div");
    row2.className = "bsai-vlp-row";

    var captureBtn = document.createElement("button");
    captureBtn.className = "bsai-vlp-btn";
    captureBtn.textContent = "📸 输出当前预览帧";
    captureBtn.title = "将视频预览当前播放位置的帧设为输出帧，并切换到单帧模式";

    row2.appendChild(captureBtn);

    // 提示
    var hint = document.createElement("div");
    hint.className = "bsai-vlp-hint";
    hint.textContent = "暂停视频到目标帧 → 点击此按钮 → 执行队列";

    panel.appendChild(header);
    panel.appendChild(row1);
    panel.appendChild(row2);
    panel.appendChild(hint);

    // ---- 状态同步 ----
    function updateBadge() {
        var w = getWidget(node, "output_mode");
        if (!w) return;
        var mode = w.value;
        if (mode === "选中单帧") {
            badge.textContent = "单帧模式";
            badge.className = "bsai-vlp-badge single";
        } else {
            badge.textContent = "完整视频";
            badge.className = "bsai-vlp-badge full";
        }
    }

    function syncFromWidget() {
        var w = getWidget(node, "frame_index");
        if (w) frameInput.value = Math.round(w.value || 0);
        updateBadge();
    }

    syncFromWidget();

    // 代理 widget callback 以同步 UI
    var modeW = getWidget(node, "output_mode");
    if (modeW) {
        var origModeCb = modeW.callback;
        modeW.callback = function(v) {
            updateBadge();
            if (typeof origModeCb === "function") {
                try { return origModeCb.apply(this, arguments); } catch(e) {}
            }
        };
    }

    var frameW = getWidget(node, "frame_index");
    if (frameW) {
        var origFrameCb = frameW.callback;
        frameW.callback = function(v) {
            frameInput.value = Math.round(v);
            if (typeof origFrameCb === "function") {
                try { return origFrameCb.apply(this, arguments); } catch(e) {}
            }
        };
    }

    // ---- 按钮事件 ----
    prevBtn.addEventListener("click", function() {
        var cur = parseInt(frameInput.value) || 0;
        var val = Math.max(0, cur - 1);
        frameInput.value = val;
        setWidgetValue(node, "frame_index", val);
    });

    nextBtn.addEventListener("click", function() {
        var cur = parseInt(frameInput.value) || 0;
        var val = cur + 1;
        frameInput.value = val;
        setWidgetValue(node, "frame_index", val);
    });

    frameInput.addEventListener("keydown", function(e) {
        if (e.key === "Enter") {
            e.preventDefault();
            var val = parseInt(frameInput.value);
            if (isNaN(val) || val < 0) val = 0;
            setWidgetValue(node, "frame_index", val);
            frameInput.blur();
        }
    });

    // ---- 核心：捕获当前预览帧 ----
    captureBtn.addEventListener("click", function() {
        var nodeEl = findNodeElement(node);
        var video = findVideoInNode(nodeEl);
        var targetFrame = 0;

        if (video && !isNaN(video.duration) && video.duration > 0) {
            var currentTime = video.currentTime;
            var fpsW = getWidget(node, "frame_rate");
            var fps = fpsW ? (fpsW.value || 30) : 30;
            targetFrame = Math.floor(currentTime * fps);
        } else {
            // 找不到视频元素就用当前 frame_index
            var fw = getWidget(node, "frame_index");
            targetFrame = fw ? Math.round(fw.value || 0) : 0;
        }

        // 安全钳位
        if (targetFrame < 0) targetFrame = 0;

        // 设置帧索引
        setWidgetValue(node, "frame_index", targetFrame);
        frameInput.value = targetFrame;
        // 切换到单帧模式
        setWidgetValue(node, "output_mode", "选中单帧");
        // 更新徽章
        updateBadge();

        // 按钮闪烁反馈
        var origText = captureBtn.textContent;
        captureBtn.textContent = "✅ 已设为第 " + targetFrame + " 帧";
        captureBtn.classList.add("flash");
        setTimeout(function() {
            captureBtn.textContent = origText;
            captureBtn.classList.remove("flash");
        }, 1800);
    });

    return panel;
}

// ============================================================
// 注入面板到节点
// ============================================================

function injectPanel(node) {
    if (!node) return;
    if (node._bsai_vlp_injected) return;
    node._bsai_vlp_injected = true;

    var attempts = 0;
    var maxAttempts = 30;

    function tryInject() {
        attempts++;
        var nodeEl = findNodeElement(node);

        if (!nodeEl) {
            if (attempts < maxAttempts) setTimeout(tryInject, 400);
            return;
        }

        // 已经注入过就跳过
        if (nodeEl.querySelector('[data-bsvlp="1"]')) return;

        var videoEl = findVideoInNode(nodeEl);
        if (videoEl && videoEl.parentNode) {
            // 找到视频的父容器，插入到视频后面
            var panel = createFramePanel(node);
            videoEl.parentNode.insertBefore(panel, videoEl.nextSibling);
            return;
        }

        // 找节点内容区
        var contentEl = nodeEl.querySelector(".content") ||
                        nodeEl.querySelector(".litegraph-node-content") ||
                        nodeEl;
        if (contentEl && attempts < maxAttempts) {
            setTimeout(tryInject, 500);
            return;
        }

        // 最后兜底：直接附加到节点元素
        if (contentEl) {
            var panel = createFramePanel(node);
            contentEl.appendChild(panel);
        }
    }

    setTimeout(tryInject, 600);
}

// ============================================================
// 扫描所有现有节点
// ============================================================

function scanAllNodes() {
    var graph = window.LiteGraph && window.LiteGraph.graph ? window.LiteGraph.graph :
                  (window.app && window.app.graph ? window.app.graph : null);
    if (!graph || !graph._nodes) return;

    for (var i = 0; i < graph._nodes.length; i++) {
        var node = graph._nodes[i];
        if (isVideoLoaderPlusNode(node)) {
            injectPanel(node);
        }
    }
}

// ============================================================
// 监听节点创建
// ============================================================

function setupNodeCreationHook() {
    var graph = window.app && window.app.graph ? window.app.graph :
                (window.LiteGraph && window.LiteGraph.graph ? window.LiteGraph.graph : null);
    if (!graph) {
        setTimeout(setupNodeCreationHook, 500);
        return;
    }

    // 钩子：节点添加时检测
    var origOnNodeAdded = graph.onNodeAdded;
    graph.onNodeAdded = function(node) {
        if (origOnNodeAdded) {
            try { origOnNodeAdded.apply(this, arguments); } catch(e) {}
        }
        if (isVideoLoaderPlusNode(node)) {
            injectPanel(node);
        }
    };

    // 也扫描已经存在的节点
    setTimeout(scanAllNodes, 1500);
}

// ============================================================
// 启动
// ============================================================

function start() {
    if (window.__bsai_vlp_started) return;
    window.__bsai_vlp_started = true;
    setupNodeCreationHook();
}

// 等待 app / LiteGraph 就绪
function waitForReady() {
    if (window.app && window.app.graph && window.app.graph._nodes) {
        start();
        return;
    }
    if (window.LiteGraph && window.LiteGraph.graph) {
        start();
        return;
    }
    setTimeout(waitForReady, 300);
}

if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", waitForReady);
} else {
    waitForReady();
}

// 额外保险：5 秒后强制扫描一次
setTimeout(scanAllNodes, 5000);
setTimeout(scanAllNodes, 10000);

} // end of __bsai_vlp_main_loaded guard
