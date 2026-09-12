# BSAI_ComfyUI_Nodes ｜ BSAI_ComfyUI_Nodes

**BSAI 自定义 ComfyUI 节点合集 / BSAI Custom Nodes Collection for ComfyUI**

> 整合视频处理、图像处理、音频处理、LLM 推理、AI 生成等多种功能节点，共 **28 个节点**。
> A collection of 28 ComfyUI nodes covering video processing, image processing, audio processing, LLM inference and AI generation.

> 中英双语文档 / Bilingual documentation.

---

## 安装 / Installation

### 方法一：ComfyUI Manager / Method 1: ComfyUI Manager
1. 打开 ComfyUI Manager → 2. "Custom Nodes Manager" → 3. 搜索 `BSAI_ComfyUI_Nodes` → 4. 安装并重启 / Install & restart

### 方法二：手动安装 / Method 2: Manual
```bash
git clone https://github.com/xm6018924/BSAI_ComfyUI_Nodes.git
pip install -r requirements.txt   # 可选依赖按需安装
```
重启 ComfyUI / Restart ComfyUI.

---

## 节点列表 / Node List

### LLM 推理 / LLM Inference

#### BSAI Qwen Nodes（基于 llama-cpp-python 本地推理 / local inference）

| 节点 / Node | 功能 / Function |
|---|---|
| **BSAI Qwen Model Loader** | 加载 Qwen3-VL / Qwen3.5-VL / Qwen3.6-VL / Qwen3.8-VL / Gemma4（GGUF），支持多模态视觉投影 / Load GGUF LLM with vision projection |
| **BSAI Qwen Prompt Inference** | 纯文本提示词推理，支持温度、top_p、top_k / Text prompt inference |
| **BSAI Qwen Multimodal Inference** | 多模态推理：5 张图片 + 3 个视频输入，自动采样视频帧 / Multimodal inference |
| **BSAI Qwen Unload Model** | 卸载模型释放显存（默认推理后自动卸载）/ Unload model to free VRAM |
| **BSAI Multiple Paths Input Plus** | 多路径合并为批量输入，图片/视频自动识别 / Merge multiple paths into a batch |
| **BSAI Video Loader Plus** | 视频加载器：帧率、起止帧、缩放等 / Video loader with fps/range/scale |

> 旧版节点 `MultiplePathsInputPlus` / `VideoLoaderPlus` 保留兼容（同功能）/ Legacy aliases kept for compatibility.

#### OLLAMA 推理 / OLLAMA Inference

| 节点 / Node | 功能 / Function |
|---|---|
| **BSAI 多图像视频反推** | 10 张图片 + 5 个视频输入，用 OLLAMA（Qwen3 / Qwen3.5 / Qwen3.6 / Qwen3.8 / Gemma4）生成描述，返回响应文本、思考过程与提示词列表 / Multi-image-video captioning via OLLAMA |

---

### AI 图像/视频生成 / AI Generation

#### Krea 2 生成 / Krea 2

| 节点 / Node | 功能 / Function |
|---|---|
| **BSAI Krea 2 Style Reference** | 参考图编码为风格条件（CONDITIONING），轻微/平衡/强烈三级强度 / Encode style reference |
| **BSAI Krea 2 Image** | 一体化生图：模型加载 + 提示词编码 + 2 个风格参考 + 采样 + VAE 解码 / All-in-one image generation |

#### LingBot-Video 生成 / LingBot-Video

| 节点 / Node | 功能 / Function |
|---|---|
| **BSAI LingBot-Video Loader (Dense 1.3B)** | 加载 LingBot-Video Dense 1.3B，支持自动从 HuggingFace 下载 / Load model (auto-download) |
| **BSAI LingBot-Video Text-to-Video** | 文生视频 T2V：自定义分辨率、帧数、采样步数 / Text-to-video |
| **BSAI LingBot-Video Text-to-Image** | 文生图 T2I / Text-to-image |
| **BSAI LingBot-Video Image-to-Video** | 图文生视频 TI2V：基于参考图生成视频 / Image-to-video |
| **BSAI LingBot-Video Unload** | 卸载模型释放显存 / Unload model |

---

### 视频处理 / Video Processing

| 节点 / Node | 功能 / Function |
|---|---|
| **BSAI PIP MultiLayer (WithAudio)** | 多视频画中画合成：4 层叠加，每层自定义 X/Y/Width/Height，4 路音频选择输出 / Multi-layer PIP compositing |
| **BSAI Video To Images** | 视频转图像序列 / Video → image sequence |
| **BSAI Image Sequence To Video** | 图像序列合成视频：CPU/GPU 编码、多格式输出、音频合并 / Image sequence → video |
| **BSAI Merge Video+Audio to Images** | 合并视频和音频后按自定义帧率拆解为图像序列 / Merge A/V then split to frames |
| **BSAI MultiImageVideoReverse** | 图像/视频批次反转（逆序）/ Reverse image/video batch order |

---

### 图像处理 / Image Processing

| 节点 / Node | 功能 / Function |
|---|---|
| **BSAI Draw Text Overlay** | 图像绘制文字：Windows 系统字体、CJK、描边、对齐 / Draw text overlay |
| **BSAI Merge Images** | 多图合并网格图（2x2/3x3/自定义），循环累积模式 / Merge images into grid |
| **Compress Images** | 压缩图像为 JPEG/PNG/WEBP，可调质量与压缩级别 / Compress images |

---

### 音频处理 / Audio Processing

| 节点 / Node | 功能 / Function |
|---|---|
| **AudioCropProcessUTK** | 裁剪音频：MM:SS 格式或秒数指定起止 / Crop audio by time |
| **BSAI Audio Duration To Frames** | 音频时长转帧数 / Audio duration → frames |
| **BSAI Audio Frames To Duration** | 帧数转音频时长 / Frames → audio duration |

---

### 工具节点 / Utility

| 节点 / Node | 功能 / Function |
|---|---|
| **BSAI Any To List** | 任意输入转列表，支持重复次数 / Convert any input to list |
| **BSAI LongTextToList** | 按分隔符分割长文本为列表，返回列表与总段数 / Split long text into list |

---

## 依赖说明 / Dependencies

- **核心 / Core**：torch, numpy, Pillow, opencv-python
- **LLM**：ollama（OLLAMA 节点）, llama-cpp-python（Qwen 节点）
- **音频 / Audio**：scipy, soundfile
- **LingBot-Video**：transformers>=5.0.0, diffusers>=0.37.0, peft, decord, json_repair, huggingface_hub, imageio
- **其他 / Others**：pydantic, packaging
- `folder_paths` / `comfy` 由 ComfyUI 提供，无需安装 / Provided by ComfyUI.

## 模型文件存放路径 / Model Paths

| 模型类型 / Model | 路径 / Path |
|---|---|
| Qwen/Gemma GGUF | `ComfyUI/models/LLM/` |
| LingBot-Video | `ComfyUI/models/lingbot-vision/dense-1.3b/` |

---

## License / 许可证

本插件代码 Apache-2.0。/ This plugin is Apache-2.0 licensed.
