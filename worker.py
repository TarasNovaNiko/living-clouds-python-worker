import os
import base64
from io import BytesIO
from typing import Any, Dict, Optional, Tuple

import runpod
import requests
from PIL import Image, UnidentifiedImageError

# ----------------------------
# Cache dirs: avoid filling root disk
# ----------------------------
os.environ.setdefault("HF_HOME", "/tmp/hf")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/tmp/hf/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/hf/transformers")
os.environ.setdefault("DIFFUSERS_CACHE", "/tmp/hf/diffusers")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# ----------------------------
# Model (SD 1.5) – lazy-loaded
# ----------------------------
PIPE_TXT2IMG = None
PIPE_IMG2IMG = None


def _load_pipelines() -> Tuple[Any, Any]:
    global PIPE_TXT2IMG, PIPE_IMG2IMG

    if PIPE_TXT2IMG is not None and PIPE_IMG2IMG is not None:
        return PIPE_TXT2IMG, PIPE_IMG2IMG

    import torch
    from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Deploy endpoint as GPU worker.")

    model_id = os.getenv("SD_MODEL_ID", "runwayml/stable-diffusion-v1-5")
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

    dtype = torch.float16
    PIPE_TXT2IMG = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
        use_auth_token=hf_token,
    ).to("cuda")

    PIPE_IMG2IMG = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
        use_auth_token=hf_token,
    ).to("cuda")

    try:
        PIPE_TXT2IMG.enable_attention_slicing()
        PIPE_IMG2IMG.enable_attention_slicing()
    except Exception:
        pass

    return PIPE_TXT2IMG, PIPE_IMG2IMG


# ----------------------------
# Image helpers
# ----------------------------
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(15 * 1024 * 1024)))  # 15 MB


def _strip_data_url_prefix(b64: str) -> str:
    if "," in b64 and b64.strip().lower().startswith("data:"):
        return b64.split(",", 1)[1]
    return b64


def _image_from_base64(b64: str) -> Image.Image:
    b64 = _strip_data_url_prefix(b64).strip()
    raw = base64.b64decode(b64, validate=True)

    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError(f"Decoded image is too large ({len(raw)} bytes). Limit is {MAX_IMAGE_BYTES} bytes.")

    try:
        img = Image.open(BytesIO(raw))
        img.load()
        return img.convert("RGB")
    except UnidentifiedImageError:
        raise ValueError(
            "Decoded init_image_base64 is not a valid image (PNG/JPG). "
            "Most common cause: the base64 string is truncated when copying/pasting."
        )


def _resolve_imbb_to_direct(url: str) -> str:
    if "ibb.co/" not in url or "i.ibb.co/" in url:
        return url

    r = requests.get(url, timeout=20)
    r.raise_for_status()

    html = r.text
    for key in ['property="og:image"', "property='og:image'"]:
        idx = html.find(key)
        if idx != -1:
            part = html[idx: idx + 500]
            for q in ['content="', "content='"]:
                j = part.find(q)
                if j != -1:
                    part2 = part[j + len(q):]
                    end = part2.find('"') if q.endswith('"') else part2.find("'")
                    if end != -1:
                        direct = part2[:end].strip()
                        if direct.startswith("http"):
                            return direct

    return url


def _download_image(url: str) -> Image.Image:
    url = url.strip()
    url = _resolve_imbb_to_direct(url)

    with requests.get(url, stream=True, timeout=30, allow_redirects=True) as r:
        r.raise_for_status()
        content = BytesIO()
        total = 0
        for chunk in r.iter_content(chunk_size=1024 * 128):
            if not chunk:
                continue
            total += len(chunk)
            if total > MAX_IMAGE_BYTES:
                raise ValueError(
                    f"Downloaded image is too large (> {MAX_IMAGE_BYTES} bytes). "
                    "Use a smaller image or increase MAX_IMAGE_BYTES."
                )
            content.write(chunk)

    content.seek(0)
    try:
        img = Image.open(content)
        img.load()
        return img.convert("RGB")
    except UnidentifiedImageError:
        raise ValueError(
            "init_image_url did not point to a valid image file (PNG/JPG). "
            "If you use imgbb, use the direct image link (starts with https://i.ibb.co/...) "
            "or paste the imgbb page link and let the worker resolve it."
        )


def _pick_size(input_data: Dict[str, Any]) -> Tuple[int, int]:
    w = int(input_data.get("width", 512))
    h = int(input_data.get("height", 512))
    w = max(64, (w // 8) * 8)
    h = max(64, (h // 8) * 8)
    return w, h


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    try:
        input_data = event.get("input", {}) or {}
        mode = (input_data.get("mode") or "txt2img").lower()
        prompt = input_data.get("prompt") or ""

        negative_prompt = input_data.get("negative_prompt") or ""
        steps = int(input_data.get("steps", 20))
        guidance = float(input_data.get("guidance", 7.5))

        if not prompt:
            return {"ok": False, "stage": "validation", "error": "prompt is required"}

        pipe_t2i, pipe_i2i = _load_pipelines()

        if mode == "txt2img":
            width, height = _pick_size(input_data)
            result = pipe_t2i(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                width=width,
                height=height,
            )
            img = result.images[0]

            buf = BytesIO()
            img.save(buf, format="PNG")
            image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            return {
                "ok": True,
                "mode": "txt2img",
                "image_base64": image_b64,
                "meta": {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "steps": steps,
                    "guidance": guidance,
                    "width": width,
                    "height": height,
                    "note": "SD1.5 via diffusers",
                },
            }

        if mode == "img2img":
            strength = float(input_data.get("strength", 0.6))

            init_img: Optional[Image.Image] = None
            if input_data.get("init_image_base64"):
                init_img = _image_from_base64(input_data["init_image_base64"])
            elif input_data.get("init_image_url"):
                init_img = _download_image(input_data["init_image_url"])
            else:
                return {"ok": False, "stage": "validation", "error": "img2img requires init_image_base64 or init_image_url"}

            result = pipe_i2i(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=init_img,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=guidance,
            )
            img = result.images[0]

            buf = BytesIO()
            img.save(buf, format="PNG")
            image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            return {
                "ok": True,
                "mode": "img2img",
                "image_base64": image_b64,
                "meta": {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "strength": strength,
                    "steps": steps,
                    "guidance": guidance,
                    "note": "SD1.5 img2img via diffusers",
                },
            }

        return {"ok": False, "stage": "validation", "error": f"Unknown mode: {mode}"}

    except Exception as e:
        return {"ok": False, "stage": "exception", "error": str(e)}


runpod.serverless.start({"handler": handler})
