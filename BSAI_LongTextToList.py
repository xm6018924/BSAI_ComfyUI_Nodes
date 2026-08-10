import json

class BSAI_LongTextToList:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "default": "",
                    "multiline": True
                }),
                "delimiter": ("STRING", {
                    "default": "\n",
                    "multiline": False
                }),
            },
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("text_list", "总段数")
    FUNCTION = "split_text"
    CATEGORY = "BSAI/Text"

    def split_text(self, text, delimiter):
        # 分割文本
        text_list = [item.strip() for item in text.split(delimiter) if item.strip()]
        # 返回分割后的列表和列表长度
        return (text_list, len(text_list))

# 注册节点
NODE_CLASS_MAPPINGS = {
    "BSAI_LongTextToList": BSAI_LongTextToList
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BSAI_LongTextToList": "BSAI LongTextToList"
}