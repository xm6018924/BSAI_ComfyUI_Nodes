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

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
