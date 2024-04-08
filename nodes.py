import torch

torch.backends.cuda.matmul.allow_tf32 = True


class CachingCLIPTextEncode:

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "conditioning"

    def __init__(self):
        self.cache: dict[str, str | torch.Tensor | None] = {
            "text": None,
            "cond": None,
            "pooled_output": None,
        }

    @classmethod
    def INPUT_TYPES(s) -> dict:
        return {
            "required": {"text": ("STRING", {"multiline": True}), "clip": ("CLIP",)}
        }

    def encode(
        self, clip: torch.nn.Module, text: str
    ) -> tuple[list[list[torch.Tensor, dict[str, torch.Tensor]]]]:
        if text != self.cache["text"]:
            tokens = clip.tokenize(text)
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
            return ([[cond, {"pooled_output": pooled}]],)
        else:
            return (
                [[self.cache["cond"], {"pooled_output": self.cache["pooled_output"]}]],
            )
