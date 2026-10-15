// BSAI Qwen Model Loader - 后端切换时动态显隐参数
// 参考 BSAI-Qwen-Prompt-Enhancer 的实现

// 公共参数：两个后端始终可见
const COMMON_WIDGETS = [
  "推理后端",
  "启用思考",
  "上下文长度",
];

// 后端值 -> 该后端可见的 widget 名
const BACKEND_WIDGET_GROUPS = {
  "llama.cpp (本地加载GGUF)": COMMON_WIDGETS.concat(["模型系列", "主模型", "视觉投影mmproj", "GPU层数"]),
  "BSAI Strata HTTP (Qwen3.8-Flash-Next @127.0.0.1:8080)": COMMON_WIDGETS.concat(["模型挡位"]),
};

function setWidgetHidden(widget, hidden) {
  if (!widget) return;
  if (widget.options) {
    widget.options.hidden = hidden;
  }
  // 新版 Vue 前端兼容
  if (widget.element) {
    widget.element.style.display = hidden ? "none" : "";
  }
}

function syncWidgetVisibility(node) {
  if (!node || !node.widgets) return;
  const backendWidget = node.widgets.find((w) => w.name === "推理后端");
  if (!backendWidget) return;

  const backendValue = backendWidget.value;
  const visibleWidgets = BACKEND_WIDGET_GROUPS[backendValue] || COMMON_WIDGETS;

  for (const w of node.widgets) {
    if (w.name === "推理后端") continue;
    setWidgetHidden(w, !visibleWidgets.includes(w.name));
  }
}

app.registerExtension({
  name: "BSAI.QwenModelLoader",
  nodeCreated(node) {
    if (node.comfyClass !== "BSAI_QwenModelLoader") return;

    // 后端切换时同步显隐
    const backendWidget = node.widgets.find((w) => w.name === "推理后端");
    if (backendWidget) {
      const origCallback = backendWidget.callback;
      backendWidget.callback = function (value, ...args) {
        const r = origCallback ? origCallback.call(this, value, ...args) : undefined;
        syncWidgetVisibility(node);
        return r;
      };
    }

    // 工作流加载时恢复显隐
    const origConfigure = node.onConfigure;
    node.onConfigure = function (...args) {
      const r = origConfigure ? origConfigure.apply(this, args) : undefined;
      setTimeout(() => syncWidgetVisibility(node), 100);
      return r;
    };

    // 初始同步
    setTimeout(() => syncWidgetVisibility(node), 100);
  },
});
