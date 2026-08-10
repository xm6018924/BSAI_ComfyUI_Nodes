import torch
import folder_paths

class BSAI_VideoToImages:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "video_to_images"
    CATEGORY = "BSAI"
    DESCRIPTION = "Converts a video to an image sequence."

    def video_to_images(self, video):
        # Get video components
        components = video.get_components()
        images = components.images
        return (images,)

NODE_CLASS_MAPPINGS = {
    "BSAI_VideoToImages": BSAI_VideoToImages,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BSAI_VideoToImages": "BSAI Video To Images",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]