import base64
import binascii
import io
import os
from typing import Any, Dict, Optional

import runpod
import torch
from PIL import Image, UnidentifiedImageError

from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline


MODEL_ID = os.getenv("MODEL_ID", "runwayml/stable-diffusion-v1-5")
DEFAULT_STEPS = int(os.getenv("DEFAULT_STEPS", "20"))
DEFAULT_GUIDANCE = float(os.getenv("DEFAULT_GUIDANCE", "7.5"))
DEFAULT_STRENGTH = float(os.getenv("DEFAULT_STRENGTH", "0.6"))

_device = "cuda" if torch.cuda.is_available() else "cpu"
_txt2img_pipe: Optional[StableDiffusionPipeline] = None
_img2img_pipe: Optional[StableDiffusionImg2ImgPipeline] = None


def _load_pipes():
    global _txt2img_pipe, _img2img_pipe

    if _device != "cuda":
        raise RuntimeError("CUDA is not available. Deploy endpoint as GPU worker.")

    if _txt2img_pipe is None:
        _txt2img_pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(_device)
        _txt2img_pipe.enable_attention_slicing()

    if _img2img_pipe is None:
        _img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(_device)
        _img2img_pipe.enable_attention_slicing()


def _normalize_base64(s: str) -> str:
    """
    - приймає або "data:image/...;base64,AAA...", або чистий base64
    - прибирає пробіли/переноси
    - додає padding '=' якщо потрібно
    """
    if not isinstance(s, str) or not s.strip():
        raise ValueError("init_image_base64 is empty or not a string")

    s = s.strip()

    # data URL -> залишаємо лише частину після коми
    if s.lower().startswith("data:") and "," in s:
        s = s.split(",", 1)[1]

    # прибираємо всі пробіли/переноси (часто з'являються при копіюванні)
    s = "".join(s.split())

    # padding
    pad = len(s) % 4
    if pad:
        s += "=" * (4 - pad)

    return s


def _b64_to_pil(image_base64: str) -> Image.Image:
    b64 = _normalize_base64(image_base64)

    try:
        data = base64.b64decode(b64, validate=False)
    except (binascii.Error, ValueError) as e:
        raise ValueError(
            "init_image_base64 is not valid base64. "
            "Make sure you paste the full string (not truncated)."
        ) from e

    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return img
    except UnidentifiedImageError as e:
        # Це саме твій кейс: base64 декодується, але байти не є PNG/JPG (часто обрізано).
        raise ValueError(
            "Decoded init_image_base64 is not a valid image (PNG/JPG). "
            "Most common cause: the base64 string is truncated when copying/pasting. "
            "Try copying from the JSON using the copy icon, and ensure the string is complete."
        ) from e


def _pil_to_b64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _resize_to_multiple_of_8(img: Image.Image) -> Image.Image:
    # SD1.5 працює стабільніше на розмірах кратних 8
    w, h = img.size
    w2 = max(8, (w // 8) * 8)
    h2 = max(8, (h // 8) * 8)
    if (w2, h2) != (w, h):
        img = img.resize((w2, h2))
    return img


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    try:
        input_data = event.get("input", {}) or {}
        mode = (input_data.get("mode") or "txt2img").lower()

        prompt = input_data.get("prompt", "")
        negative_prompt = input_data.get("negative_prompt", "")

        steps = int(input_data.get("steps", DEFAULT_STEPS))
        guidance = float(input_data.get("guidance", DEFAULT_GUIDANCE))
        seed = input_data.get("seed", None)

        generator = torch.Generator(device=_device).manual_seed(int(seed)) if seed is not None else None

        _load_pipes()

        if mode == "txt2img":
            out = _txt2img_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
            )
            img = out.images[0]
            return {
                "ok": True,
                "mode": "txt2img",
                "image_base64": _pil_to_b64_png(img),
                "meta": {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "steps": steps,
                    "guidance": guidance,
                    "seed": seed,
                    "model": MODEL_ID,
                },
            }

        if mode == "img2img":
            init_b64 = input_data.get("init_image_base64")
            if not init_b64:
                return {
                    "ok": False,
                    "stage": "validation_error",
                    "error": "init_image_base64 is required for img2img",
                }

            strength = float(input_data.get("strength", DEFAULT_STRENGTH))

            init_img = _b64_to_pil(init_b64)
            init_img = _resize_to_multiple_of_8(init_img)

            out = _img2img_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=init_img,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
            )
            img = out.images[0]
            return {
                "ok": True,
                "mode": "img2img",
                "image_base64": _pil_to_b64_png(img),
                "meta": {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "strength": strength,
                    "steps": steps,
                    "guidance": guidance,
                    "seed": seed,
                    "model": MODEL_ID,
                    "init_size": list(init_img.size),
                },
            }

        return {"ok": False, "stage": "unknown_mode", "error": f"Unknown mode: {mode}"}

    except Exception as e:
        return {"ok": False, "stage": "exception", "error": str(e)}


runpod.serverless.start({"handler": handler})
