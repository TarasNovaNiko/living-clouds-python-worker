import os
import io
import base64
from typing import Optional, Dict, Any

import runpod
import torch
from PIL import Image

from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline


MODEL_ID = "runwayml/stable-diffusion-v1-5"

_txt2img_pipe: Optional[StableDiffusionPipeline] = None
_img2img_pipe: Optional[StableDiffusionImg2ImgPipeline] = None


def _disable_safety(pipe):
    # Вимикаємо safety_checker, щоб не блокував нормальні зображення.
    def dummy(images, **kwargs):
        return images, [False] * len(images)
    pipe.safety_checker = dummy
    return pipe


def _get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_pipes():
    global _txt2img_pipe, _img2img_pipe

    if _txt2img_pipe is not None and _img2img_pipe is not None:
        return

    device = _get_device()
    if device != "cuda":
        # На CPU SD1.5 буде дуже повільний для serverless.
        raise RuntimeError("CUDA is not available. Deploy endpoint as GPU worker.")

    torch_dtype = torch.float16

    # txt2img
    _txt2img_pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch_dtype,
        safety_checker=None,  # додатково вимкнемо нижче
    )
    _txt2img_pipe = _disable_safety(_txt2img_pipe)
    _txt2img_pipe = _txt2img_pipe.to(device)
    _txt2img_pipe.enable_attention_slicing()

    # img2img
    _img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch_dtype,
        safety_checker=None,
    )
    _img2img_pipe = _disable_safety(_img2img_pipe)
    _img2img_pipe = _img2img_pipe.to(device)
    _img2img_pipe.enable_attention_slicing()


def _b64_to_pil(b64_str: str) -> Image.Image:
    data = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return img


def _pil_to_b64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _make_generator(seed: Optional[int], device: str):
    if seed is None:
        return None
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    return g


def handler(event: Dict[str, Any]):
    try:
        _load_pipes()
    except Exception as e:
        return {"ok": False, "error": str(e), "stage": "load_model_failed"}

    device = _get_device()

    input_data = event.get("input", {}) or {}
    mode = (input_data.get("mode") or "txt2img").lower()

    prompt = input_data.get("prompt", "")
    negative_prompt = input_data.get("negative_prompt", "") or None

    steps = int(input_data.get("steps", 25))
    guidance = float(input_data.get("guidance", 7.0))
    seed = input_data.get("seed", None)

    generator = _make_generator(seed, device)

    try:
        if mode == "txt2img":
            width = int(input_data.get("width", 512))
            height = int(input_data.get("height", 512))

            out = _txt2img_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                width=width,
                height=height,
                generator=generator,
            )
            img = out.images[0]
            return {
                "ok": True,
                "mode": "txt2img",
                "image_base64": _pil_to_b64_png(img),
                "meta": {
                    "model": MODEL_ID,
                    "steps": steps,
                    "guidance": guidance,
                    "seed": seed,
                    "width": width,
                    "height": height,
                },
            }

        if mode == "img2img":
            init_b64 = input_data.get("init_image_base64")
            if not init_b64:
                return {"ok": False, "error": "init_image_base64 is required for img2img"}

            strength = float(input_data.get("strength", 0.6))
            init_img = _b64_to_pil(init_b64)

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
                    "model": MODEL_ID,
                    "steps": steps,
                    "guidance": guidance,
                    "seed": seed,
                    "strength": strength,
                },
            }

        return {"ok": False, "error": f"Unknown mode: {mode}"}

    except Exception as e:
        return {"ok": False, "error": str(e), "stage": "inference_failed"}


runpod.serverless.start({"handler": handler})
