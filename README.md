# BSAI_ComfyUI_Nodes

BSAI 自定义 ComfyUI 节点合集，整合了视频处理、图像处理、音频处理、LLM 推理、AI 生成等多种功能节点。

## 安装

### 方法一：通过 ComfyUI Manager 安装
1. 打开 ComfyUI Manager
2. 选择 "Custom Nodes Manager"
3. 搜索 "BSAI_ComfyUI_Nodes"
4. 点击安装并重启 ComfyUI

### 方法二：手动安装
1. 进入 ComfyUI 的 `custom_nodes` 目录
2. 克隆仓库：
   ```bash
   git clone https://github.com/xm6018924/BSAI_ComfyUI_Nodes.git
   ```
3. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
4. 重启 ComfyUI

## 节点列表

本插件包含以下 **25 个节点**，按功能分类：

---

### LLM 推理

#### BSAI Qwen Nodes（基于 llama-cpp-python 本地推理）

| 节点名称 | 功能说明 |
|---------|---------|
| **BSAI Qwen Model Loader** | 加载 Qwen3-VL / Qwen3.5-VL / Qwen3.6-VL / Qwen3.8-VL / Gemma4 模型（GGUF 格式），支持多模态视觉投影 |
| **BSAI Qwen Prompt Inference** | 纯文本提示词推理，支持温度、top_p、top_k 等参数调节 |
| **BSAI Qwen Multimodal Inference** | 多模态推理，支持 5 张图片 + 3 个视频输入，自动采样视频帧 |
| **BSAI Qwen Unload Model** | 卸载模型释放显存 |
| **BSAI Multiple Paths Input Plus** | 多路径合并为批量输入，支持图片和视频格式自动识别 |
| **BSAI Video Loader Plus** | 视频加载器，支持帧率、起止帧、缩放等参数 |

#### OLLAMA 推理

| 节点名称 | 功能说明 |
|---------|---------|
| **BSAI 多图像视频反推** | 支持 10 张图片 + 5 个视频输入，使用 OLLAMA 模型（Qwen3 / Qwen3.5 / Qwen3.6 / Qwen3.8 / Gemma4）分析并生成描述，返回响应文本、思考过程和提示词列表 |

---

### AI 图像/视频生成

#### Krea 2 生成

| 节点名称 | 功能说明 |
|---------|---------|
| **BSAI Krea 2 Style Reference** | 将参考图编码为风格条件（CONDITIONING），支持轻微/平衡/强烈三级风格强度 |
| **BSAI Krea 2 Image** | 一体化生图节点，集成模型加载、提示词编码、2 个风格参考、采样、VAE 解码全流程 |

#### LingBot-Video 生成

| 节点名称 | 功能说明 |
|---------|---------|
| **BSAI LingBot-Video Loader (Dense 1.3B)** | 加载 LingBot-Video Dense 1.3B 模型，支持自动从 HuggingFace 下载 |
| **BSAI LingBot-Video Text-to-Video** | 文生视频（T2V），支持自定义分辨率、帧数、采样步数 |
| **BSAI LingBot-Video Text-to-Image** | 文生图（T2I） |
| **BSAI LingBot-Video Image-to-Video** | 图文生视频（TI2V），基于参考图生成视频 |
| **BSAI LingBot-Video Unload** | 卸载模型释放显存 |

---

### 视频处理

| 节点名称 | 功能说明 |
|---------|---------|
| **BSAI PIP MultiLayer** | 多视频画中画合成，支持 4 层视频叠加，每层自定义 X/Y/Width/Height，支持 4 路音频选择输出 |
| **BSAI Video To Images** | 将视频转换为图像序列 |
| **BSAI Image Sequence To Video** | 将图像序列合成为视频，支持 CPU/GPU 编码、多格式输出、音频合并 |
| **BSAI - Merge Video+Audio to Images** | 合并视频和音频后按自定义帧率拆解为图像序列 |

---

### 图像处理

| 节点名称 | 功能说明 |
|---------|---------|
| **BSAI Draw Text Overlay** | 在图像上绘制文字覆盖层，支持 Windows 系统字体选择、CJK 文字、描边、对齐 |
| **BSAI Merge Images** | 将多张图片合并为网格图（2x2/3x3/自定义），支持循环累积模式 |
| **Compress Images** | 压缩图像为 JPEG/PNG/WEBP 格式，可调节质量和压缩级别 |

---

### 音频处理

| 节点名称 | 功能说明 |
|---------|---------|
| **AudioCropProcessUTK** | 裁剪音频，支持 MM:SS 格式或秒数指定起止时间 |
| **BSAI Audio Duration To Frames** | 音频时长转帧数 |
| **BSAI Audio Frames To Duration** | 帧数转音频时长 |

---

### 工具节点

| 节点名称 | 功能说明 |
|---------|---------|
| **BSAI Any To List** | 将任意输入转换为列表格式，支持重复次数设置 |
| **BSAI LongTextToList** | 按分隔符将长文本分割为列表，返回列表和总段数 |

---

## 依赖说明

- **核心依赖**：torch, numpy, Pillow, opencv-python
- **LLM 推理**：ollama（OLLAMA 节点）, llama-cpp-python（Qwen 节点）
- **音频处理**：scipy, soundfile
- **LingBot-Video**：transformers>=5.0.0, diffusers>=0.37.0, peft, decord, json_repair, huggingface_hub, imageio
- **其他**：pydantic, packaging

> 注意：`folder_paths` 和 `comfy` 模块由 ComfyUI 提供，无需单独安装。

## 模型文件存放路径

| 模型类型 | 存放路径 |
|---------|---------|
| Qwen/Gemma GGUF 模型 | `ComfyUI/models/LLM/` |
| LingBot-Video 模型 | `ComfyUI/models/lingbot-vision/dense-1.3b/` |

## 已移除的重复节点

以下重复或功能相似的节点已从本插件中移除：

| 已移除文件 | 原因 | 保留的替代节点 |
|-----------|------|--------------|
| `BSAI-comfyui_pip_multilayer.py` | 与 BSAI_PIPMultiLayer.py 代码完全相同 | BSAI_PIPMultiLayer.py |
| `BSAI-video_pip_node.py` | 与 BSAI_VideoPIP.py 代码完全相同，且功能不如 PIPMultiLayer | BSAI_PIPMultiLayer.py |
| `BSAI_VideoPIP.py` | 与 BSAI-video_pip_node.py 代码完全相同 | BSAI_PIPMultiLayer.py |
| `BSAI_VideoToPrompt.py` | 与 MultiImageReverse.py 功能相似，但功能更少 | MultiImageReverse.py |
| `BSAIVideoToPrompt.py` | 与 MultiImageReverse.py 功能相似，但功能更少 | MultiImageReverse.py |
| `BSAI_MultimodalProcessor.py` | 非功能性存根节点，无实际处理逻辑 | — |
| `BSAI_PT_H3_AVLatent.py` | 属于 BSAI-MiniMAX-H3-Prompt 插件 | 见 BSAI-MiniMAX-H3-Prompt 仓库 |

## 许可证

MIT License

## 作者

BSAI Team
