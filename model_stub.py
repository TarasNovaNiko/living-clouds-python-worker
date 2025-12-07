# living-clouds-python/model_stub.py

"""
Шар взаємодії з моделлю.

Тут дві функції:
- generate_txt2img(prompt: str) -> dict
- generate_img2img(prompt: str, image_base64: str) -> dict

Зараз:
- txt2img намагається використати Stable Diffusion через diffusers.
- img2img поки лишається заглушкою (ми підключимо його пізніше).
"""

import base64
import io
from typing import Optional


from PIL import Image
import os  # <- додали

USE_SD = os.getenv("USE_SD", "0") == "1"

# Спробуємо імпортувати diffusers/torch.
# Якщо не вийде (нема бібліотек / проблеми з пам'яттю) – працюємо в режимі заглушки.
try:
    import torch
    from diffusers import StableDiffusionPipeline

    _DIFFUSERS_AVAILABLE = True
except Exception:
    _DIFFUSERS_AVAILABLE = False


# Глобальна змінна для pipeline (ініціалізуємо ліниво один раз)
_txt2img_pipe: Optional["StableDiffusionPipeline"] = None


def _init_txt2img_pipeline() -> Optional["StableDiffusionPipeline"]:
    """
    Лінива ініціалізація StableDiffusionPipeline.

    Викликаємо тільки при першому запиті, щоб не вантажити модель завчасно.
    """
    global _txt2img_pipe

    if _txt2img_pipe is not None:
        return _txt2img_pipe

    if not _DIFFUSERS_AVAILABLE or not USE_SD:
        return None

    try:
        # Базова модель txt2img (v1.5). Для RunPod потім, можливо, замінимо на щось інше.
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )

        # Якщо є GPU – використовуємо його. Якщо ні – залишаємо на CPU (повільно, але працює).
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = pipe.to(device)

        # Не обов'язково, але трохи економить пам'ять
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()

        _txt2img_pipe = pipe
        return _txt2img_pipe
    except Exception as e:
        # Якщо при ініціалізації щось пішло не так – не валимо сервіс, а працюємо в режимі заглушки.
        print(f"[model_stub] Не вдалося ініціалізувати StableDiffusionPipeline: {e}")
        _txt2img_pipe = None
        return None


def _pil_image_to_base64(img: Image.Image) -> str:
    """
    Конвертує PIL.Image у base64-рядок (PNG).
    """
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def generate_txt2img(prompt: str) -> dict:
    """
    Спробувати згенерувати зображення через Stable Diffusion.
    Якщо не вийде (нема моделі / помилка) – повернути фейкову відповідь.
    """
    pipe = _init_txt2img_pipeline()

    # Якщо pipeline недоступний – працюємо в режимі заглушки.
    if pipe is None:
        fake_base64 = "FAKE_BASE64_TXT2IMG_IMAGE"
        return {
            "ok": False,
            "mode": "txt2img",
            "prompt": prompt,
            "image_base64": fake_base64,
            "error": "StableDiffusionPipeline не доступний, використовується заглушка.",
        }

    try:
        # Можемо додати прості параметри типу num_inference_steps, guidance_scale тощо.
        result = pipe(
            prompt,
            num_inference_steps=25,
            guidance_scale=7.5,
        )

        img = result.images[0]
        img_b64 = _pil_image_to_base64(img)

        return {
            "ok": True,
            "mode": "txt2img",
            "prompt": prompt,
            "image_base64": img_b64,
        }
    except Exception as e:
        # Якщо генерація впала – не валимо сервіс, а повертаємо помилку + фейковий base64
        fake_base64 = "FAKE_BASE64_TXT2IMG_IMAGE"
        return {
            "ok": False,
            "mode": "txt2img",
            "prompt": prompt,
            "image_base64": fake_base64,
            "error": f"Помилка під час генерації: {e}",
        }


def generate_img2img(prompt: str, image_base64: str) -> dict:
    """
    Тимчасова заглушка для img2img.
    Пізніше тут буде StableDiffusionImg2ImgPipeline.
    """
    fake_result_base64 = "FAKE_BASE64_IMG2IMG_IMAGE"

    return {
        "ok": True,
        "mode": "img2img",
        "prompt": prompt,
        "input_image_base64_length": len(image_base64),
        "image_base64": fake_result_base64,
        "note": "Img2img ще не реалізований, повертається заглушка.",
    }
