import base64
import io
import os
from typing import Any, Dict, Optional

import runpod
from PIL import Image

# Optional, but strongly recommended for init_image_url
import requests

import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline


# ----------------------------
# Globals (cached per worker)
# ----------------------------
MODEL_ID = os.getenv("MODEL_ID", "runwayml/stable-diffusion-v1-5")

_txt2img_pipe: Optional[StableDiffusionPipeline] = None
_img2img_pipe: Optional[StableDiffusionImg2ImgPipeline] = None


def _disable_safety_checker(pipe):
    def dummy(images, **kwargs):
        return images, [False] * len(images)
    pipe.safety_checker = dummy
    return pipe


def _load_pipelines() -> None:
    """
    Lazy-load pipelines once per worker container.
    """
    global _txt2img_pipe, _img2img_pipe

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Deploy endpoint as GPU worker.")

    if _txt2img_pipe is None:
        _txt2img_pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
        ).to("cuda")
        _disable_safety_checker(_txt2img_pipe)
        _txt2img_pipe.enable_attention_slicing()

    if _img2img_pipe is None:
        _img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
        ).to("cuda")
        _disable_safety_checker(_img2img_pipe)
        _img2img_pipe.enable_attention_slicing()


def _strip_data_url(b64: str) -> str:
    # Accept: "data:image/png;base64,AAAA..."
    if "," in b64 and b64.strip().lower().startswith("data:"):
        return b64.split(",", 1)[1]
    return b64


def _safe_b64decode(b64: str) -> bytes:
    """
    More robust base64 decode:
    - trims whitespace/newlines
    - strips data URL prefix
    - fixes missing padding
    """
    s = _strip_data_url(b64)
    s = "".join(s.split())  # remove all whitespace/newlines

    # fix padding
    pad = len(s) % 4
    if pad:
        s += "=" * (4 - pad)

    return base64.b64decode(s, validate=False)


def _load_init_image(input_data: Dict[str, Any]) -> Image.Image:
    """
    Prefer init_image_url. If absent, use init_image_base64.
    """
    init_url = input_data.get("init_image_url")
    init_b64 = input_data.get("init_image_base64")

    if init_url:
        r = requests.get(init_url, timeout=30)
        r.raise_for_status()
        img_bytes = r.content
    elif init_b64:
        img_bytes = _safe_b64decode(init_b64)
    else:
        raise ValueError("img2img requires init_image_url or init_image_base64")

    # Validate image bytes
    try:
        img = Image.open(io.BytesIO(img_bytes))
        img.verify()  # verifies headers (doesn't decode fully)
    except Exception as e:
        raise ValueError(
            "Decoded init image is not a valid PNG/JPG. "
            "If using base64, it is likely truncated. Prefer init_image_url."
        ) from e

    # Re-open after verify
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return img


def _image_to_base64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod serverless handler.
    Expected payload:
      { "input": { "mode": "txt2img"|"img2img", ... } }
    """
    try:
        input_data = event.get("input", {}) or {}
        mode = (input_data.get("mode") or "txt2img").lower()

        _load_pipelines()

        prompt = input_data.get("prompt", "")
        negative_prompt = input_data.get("negative_prompt", "") or None
        steps = int(input_data.get("steps", 20))
        guidance = float(input_data.get("guidance", 7.5))

        if mode == "txt2img":
            result = _txt2img_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
            )
            img = result.images[0]
            return {
                "ok": True,
                "mode": "txt2img",
                "image_base64": _image_to_base64_png(img),
                "meta": {
                    "model_id": MODEL_ID,
                    "prompt": prompt,
                    "negative_prompt": negative_prompt or "",
                    "steps": steps,
                    "guidance": guidance,
                },
            }

        if mode == "img2img":
            init_img = _load_init_image(input_data)
            strength = float(input_data.get("strength", 0.6))

            result = _img2img_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=init_img,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=guidance,
            )
            img = result.images[0]
            return {
                "ok": True,
                "mode": "img2img",
                "image_base64": _image_to_base64_png(img),
                "meta": {
                    "model_id": MODEL_ID,
                    "prompt": prompt,
                    "negative_prompt": negative_prompt or "",
                    "steps": steps,
                    "guidance": guidance,
                    "strength": strength,
                    "used_init_image_url": bool(input_data.get("init_image_url")),
                },
            }

        return {"ok": False, "stage": "bad_request", "error": f"Unknown mode: {mode}"}

    except Exception as e:
        return {"ok": False, "stage": "exception", "error": str(e)}


runpod.serverless.start({"handler": handler})
