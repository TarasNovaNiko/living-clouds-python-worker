import runpod


def _num(value, default):
    try:
        return float(value)
    except Exception:
        return default


def handler(event):
    """
    RunPod Serverless handler.
    Очікує event = {"input": {...}}.

    Підтримувані режими:
      - txt2img: {"mode":"txt2img","prompt":"..."}
      - img2img: {"mode":"img2img","prompt":"...","image_base64":"...","strength":0.6}
      - echo (fallback): {"prompt":"..."}
    """
    input_data = event.get("input") or {}
    mode = (input_data.get("mode") or "txt2img").lower().strip()

    prompt = input_data.get("prompt") or ""
    negative_prompt = input_data.get("negative_prompt") or input_data.get("negativePrompt") or ""
    strength = _num(input_data.get("strength"), 0.65)

    # ---------- STUB logic ----------
    if mode == "img2img":
        image_base64 = input_data.get("image_base64") or input_data.get("imageBase64") or ""
        if not image_base64:
            return {
                "ok": False,
                "mode": "img2img",
                "error": "Missing image_base64",
                "received": {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "strength": strength,
                },
            }

        return {
            "ok": True,
            "mode": "img2img",
            "image_base64": "FAKE_BASE64_IMG2IMG_IMAGE",
            "meta": {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "strength": strength,
                "note": "Stub response. Replace with real diffusers pipeline later.",
            },
        }

    if mode == "txt2img":
        return {
            "ok": True,
            "mode": "txt2img",
            "image_base64": "FAKE_BASE64_TXT2IMG_IMAGE",
            "meta": {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "note": "Stub response. Replace with real diffusers pipeline later.",
            },
        }

    # Fallback: echo
    return {
        "ok": True,
        "mode": "echo",
        "received_prompt": prompt,
        "raw_event": event,
        "meta": {"note": "Unknown mode, returned echo"},
    }


runpod.serverless.start({"handler": handler})
