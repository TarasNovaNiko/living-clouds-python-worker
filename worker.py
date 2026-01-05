import os
import re
import time
import base64
from io import BytesIO
from typing import Any, Dict, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from PIL import Image
import runpod

import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline


# ----------------------------
# Cache / temp dirs (critical on serverless)
# ----------------------------
os.environ.setdefault("HF_HOME", "/tmp/hf")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/tmp/hf/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/hf/transformers")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")


# ----------------------------
# Model config
# ----------------------------
MODEL_ID = os.getenv("MODEL_ID", "runwayml/stable-diffusion-v1-5")

# Globals for warm reuse (container reuse between requests)
_txt2img_pipe: Optional[StableDiffusionPipeline] = None
_img2img_pipe: Optional[StableDiffusionImg2ImgPipeline] = None


# ----------------------------
# HTTP session with retries (important for 503/502/504 from image hosts)
# ----------------------------
def _make_http_session() -> requests.Session:
    s = requests.Session()

    retry = Retry(
        total=6,
        connect=6,
        read=6,
        status=6,
        backoff_factor=1.2,  # exponential-ish backoff: 0s, 1.2s, 2.4s, 4.8s...
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD"),
        raise_on_status=False,
        respect_retry_after_header=True,
    )

    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
    s.mount("http://", adapter)
    s.mount("https://", adapter)

    # “Browser-like” headers reduce CDN anti-bot false positives.
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (RunPodWorker/1.0; +https://runpod.io)",
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
    })
    return s


_http = _make_http_session()


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Deploy endpoint as GPU worker.")


def _get_txt2img_pipe() -> StableDiffusionPipeline:
    global _txt2img_pipe
    if _txt2img_pipe is None:
        _require_cuda()
        _txt2img_pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            safety_checker=None,  # MVP: disable safety checker for speed (you can re-enable later)
        ).to("cuda")
        _txt2img_pipe.enable_attention_slicing()
    return _txt2img_pipe


def _get_img2img_pipe() -> StableDiffusionImg2ImgPipeline:
    global _img2img_pipe
    if _img2img_pipe is None:
        _require_cuda()
        _img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to("cuda")
        _img2img_pipe.enable_attention_slicing()
    return _img2img_pipe


def _pil_to_base64_png(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _decode_base64_to_pil(b64: str) -> Image.Image:
    # allow "data:image/png;base64,...."
    if "," in b64 and "base64" in b64[:80]:
        b64 = b64.split(",", 1)[1]
    raw = base64.b64decode(b64)
    img = Image.open(BytesIO(raw))
    img.load()
    return img.convert("RGB")


def _extract_direct_ibb_url(html: str) -> Optional[str]:
    # imgbb pages usually contain i.ibb.co direct image
    m = re.search(r"(https://i\.ibb\.co/[^\s\"']+\.(?:png|jpg|jpeg|webp))", html, re.IGNORECASE)
    return m.group(1) if m else None


def _download_image_as_pil(url: str) -> Image.Image:
    """
    Robust downloader for init image.
    Handles:
      - ibb.co page links -> resolves to i.ibb.co direct
      - transient CDN errors (429/5xx) with retries
      - validates response is an image
    """
    # 1) If user passes ibb.co page, resolve to i.ibb.co direct image
    if "ibb.co/" in url and "i.ibb.co/" not in url:
        r = _http.get(url, timeout=30)
        if r.status_code >= 400:
            raise RuntimeError(f"Failed to open ibb.co page: HTTP {r.status_code}")
        direct = _extract_direct_ibb_url(r.text)
        if direct:
            url = direct

    # 2) Download with retries (handled by session adapter)
    r = _http.get(url, timeout=60)

    # If still 503 after retries, give actionable error.
    if r.status_code == 503:
        raise RuntimeError(
            f"503 Server Error: Service Temporarily Unavailable for url: {url}. "
            "This is usually a temporary CDN issue or rate-limit from the image host. "
            "Try: (1) re-try same request later, (2) use a different direct i.ibb.co link, "
            "(3) pass init_image_base64 instead of URL, or (4) host the image on another CDN."
        )

    if r.status_code >= 400:
        raise RuntimeError(f"Image download failed: HTTP {r.status_code} for url: {url}")

    content_type = (r.headers.get("Content-Type") or "").lower()

    # Sometimes CDNs return HTML (error page) with 200 OK
    if "image" not in content_type:
        # If it looks like HTML and contains a direct link, attempt resolve again
        text = ""
        try:
            text = r.text[:200_000]
        except Exception:
            text = ""
        direct = _extract_direct_ibb_url(text) if text else None
        if direct and direct != url:
            r2 = _http.get(direct, timeout=60)
            if r2.status_code >= 400:
                raise RuntimeError(f"Resolved direct URL failed: HTTP {r2.status_code} for url: {direct}")
            ct2 = (r2.headers.get("Content-Type") or "").lower()
            if "image" not in ct2:
                raise RuntimeError(f"Resolved URL did not return an image. Content-Type={ct2}")
            img = Image.open(BytesIO(r2.content))
            img.load()
            return img.convert("RGB")

        raise RuntimeError(f"URL did not return an image. Content-Type={content_type}")

    img = Image.open(BytesIO(r.content))
    img.load()
    return img.convert("RGB")


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    try:
        input_data = event.get("input", {}) or {}
        mode = (input_data.get("mode") or "txt2img").lower()

        prompt = input_data.get("prompt", "") or ""
        negative_prompt = input_data.get("negative_prompt", "") or ""

        steps = int(input_data.get("steps", 20))
        guidance = float(input_data.get("guidance", 7.5))
        width = int(input_data.get("width", 512))
        height = int(input_data.get("height", 512))

        if mode == "txt2img":
            pipe = _get_txt2img_pipe()
            out = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                num_inference_steps=steps,
                guidance_scale=guidance,
                width=width,
                height=height,
            )
            img = out.images[0]
            return {
                "ok": True,
                "mode": "txt2img",
                "image_base64": _pil_to_base64_png(img),
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
            pipe = _get_img2img_pipe()
            strength = float(input_data.get("strength", 0.6))

            init_img: Optional[Image.Image] = None
            if input_data.get("init_image_url"):
                init_img = _download_image_as_pil(str(input_data["init_image_url"]))
            elif input_data.get("init_image_base64"):
                init_img = _decode_base64_to_pil(str(input_data["init_image_base64"]))

            if init_img is None:
                raise ValueError("img2img requires init_image_url or init_image_base64")

            # Resize to requested size for predictable output
            init_img = init_img.resize((width, height))

            out = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                image=init_img,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=guidance,
            )
            img = out.images[0]
            return {
                "ok": True,
                "mode": "img2img",
                "image_base64": _pil_to_base64_png(img),
                "meta": {
                    "model": MODEL_ID,
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "strength": strength,
                    "steps": steps,
                    "guidance": guidance,
                    "width": width,
                    "height": height,
                },
            }

        raise ValueError(f"Unknown mode: {mode}")

    except Exception as e:
        return {
            "ok": False,
            "stage": "exception",
            "error": str(e),
        }


runpod.serverless.start({"handler": handler})
