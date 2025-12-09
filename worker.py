# worker.py
"""
RunPod serverless worker для Living Clouds.
Бере job.input з RunPod, викликає нашу логіку генерації
та повертає JSON з image_base64 (поки що – заглушка).
"""

import runpod

from model_stub import generate_txt2img_stub, generate_img2img_stub


def handler(job: dict) -> dict:
    """
    Головна функція, яку викликає RunPod.

    job: словник з ключем "input":
      {
        "input": {
          "mode": "txt2img" | "img2img",
          "prompt": "текст",
          "image_base64": "...." | null
        }
      }
    """
    data = job.get("input") or {}

    mode = data.get("mode", "txt2img")
    prompt = data.get("prompt", "") or ""
    image_base64 = data.get("image_base64")

    # Базова валідація
    if not isinstance(prompt, str) or not prompt.strip():
        return {
            "ok": False,
            "mode": mode,
            "prompt": prompt,
            "image_base64": None,
            "error": "Missing or empty 'prompt'.",
        }

    # Режим TXT2IMG (заглушка)
    if mode == "txt2img":
        result = generate_txt2img_stub(prompt)
        # очікуємо, що stub вже повертає правильний JSON
        return result

    # Режим IMG2IMG (заглушка)
    if mode == "img2img":
        if not image_base64:
            return {
                "ok": False,
                "mode": mode,
                "prompt": prompt,
                "image_base64": None,
                "error": "Missing 'image_base64' for img2img.",
            }
        result = generate_img2img_stub(image_base64, prompt)
        return result

    # Невідомий режим
    return {
        "ok": False,
        "mode": mode,
        "prompt": prompt,
        "image_base64": None,
        "error": f"Unknown mode: {mode}",
    }


if __name__ == "__main__":
    # Точка входу для RunPod serverless
    runpod.serverless.start({"handler": handler})
