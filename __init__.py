from .nodes import CachingCLIPTextEncode

NODE_CLASS_MAPPINGS = {
    "CachingCLIPTextEncode": CachingCLIPTextEncode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CachingCLIPTextEncode": "CLIP Text Encode with Caching",
}
