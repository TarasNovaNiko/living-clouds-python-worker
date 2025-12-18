import os
import base64
from io import BytesIO
from typing import Any, Dict, Optional

import requests
from PIL import Image

import runpod

# IMPORTANT: force HF caches into /tmp (helps prevent "No space left on device")
os.environ.setdefault("HF_HOME", "/tmp/hf")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/hf/transformers")
os.environ.setdefault("DIFFUSERS_CACHE", "/tmp/hf/diffusers")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

MODEL_ID = os.getenv("SD_MODEL_ID", "runwayml/stable-diffusion-v1-5")

# Optional HF token (if the model is gated on HF for your account)
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

# Lazy-loaded pipelines
_txt2img_pipe = None
_img2img_pipe = None


def _get_pipes():
    global _txt2img_pipe, _img2img_pipe

    import torch
    from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Deploy endpoint as GPU worker.")

    if _txt2img_pipe is None:
        kwargs = dict(
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
        )
        if HF_TOKEN:
            kwargs["token"] = HF_TOKEN

        _txt2img_pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, **kwargs).to("cuda")
        _txt2img_pipe.enable_attention_slicing()

        # Create img2img pipe reusing components (faster + less memory)
        _img2img_pipe = StableDiffusionImg2ImgPipeline(**_txt2img_pipe.components).to("cuda")
        _img2img_pipe.enable_attention_slicing()

    return _txt2img_pipe, _img2img_pipe


def _download_image(url: str) -> Image.Image:
    r = requests.get(url, timeout=30)
    r.raise_for_status()

    # If user accidentally provides a page URL (HTML), fail with a clear message
    content_type = (r.headers.get("content-type") or "").lower()
    if "text/html" in content_type:
        raise ValueError(
            "init_image_url returned HTML, not an image. "
            "Use a direct image URL like https://i.ibb.co/.../file.jpg"
        )

    img = Image.open(BytesIO(r.content)).convert("RGB")
    return img


def _decode_base64_image(b64: str) -> Image.Image:
    # Support data URLs: data:image/jpeg;base64,....
    if b64.startswith("data:"):
        b64 = b64.split(",", 1)[1]

    raw = base64.b64decode(b64)
    img = Image.open(BytesIO(raw)).convert("RGB")
    return img


def _image_to_base64_png(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    try:
        input_data = event.get("input", {}) or {}

        mode = (input_data.get("mode") or "txt2img").strip().lower()
        prompt = (input_data.get("prompt") or "").strip()
        negative_prompt = (input_data.get("negative_prompt") or "").strip()

        steps = int(input_data.get("steps") or 20)
        guidance = float(input_data.get("guidance") or 7.5)

        width = int(input_data.get("width") or 512)
        height = int(input_data.get("height") or 512)

        strength = float(input_data.get("strength") or 0.6)  # img2img

        if not prompt:
            return {"ok": False, "stage": "validation", "error": "prompt is required"}

        txt2img, img2img = _get_pipes()

        if mode == "txt2img":
            result = txt2img(
                prompt=prompt,
                negative_prompt=negative_prompt or None,
                num_inference_steps=steps,
                guidance_scale=guidance,
                width=width,
                height=height,
            )
            image = result.images[0]
            return {
                "ok": True,
                "mode": "txt2img",
                "image_base64": _image_to_base64_png(image),
                "meta": {
                    "model": MODEL_ID,
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "steps": steps,
                    "guidance": guidance,
                    "width": width,
                    "height": height,
                },
            }

        if mode == "img2img":
            init_image: Optional[Image.Image] = None

            init_image_url = (input_data.get("init_image_url") or "").strip()
            init_image_base64 = (input_data.get("init_image_base64") or "").strip()

            if init_image_url:
                init_image = _download_image(init_image_url)
            elif init_image_base64:
                init_image = _decode_base64_image(init_image_base64)
            else:
                return {
                    "ok": False,
                    "stage": "validation",
                    "error": "For img2img provide init_image_url (recommended) or init_image_base64",
                }

            # SD1.5 expects 512-ish. We resize safely.
            init_image = init_image.resize((width, height))

            result = img2img(
                prompt=prompt,
                negative_prompt=negative_prompt or None,
                image=init_image,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=guidance,
            )
            image = result.images[0]
            return {
                "ok": True,
                "mode": "img2img",
                "image_base64": _image_to_base64_png(image),
                "meta": {
                    "model": MODEL_ID,
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "strength": strength,
                    "steps": steps,
                    "guidance": guidance,
                    "width": width,
                    "height": height,
                    "init_image_url_used": bool(init_image_url),
                },
            }

        return {"ok": False, "stage": "validation", "error": f"Unknown mode: {mode}"}

    except Exception as e:
        return {"ok": False, "stage": "exception", "error": str(e)}


runpod.serverless.start({"handler": handler})
