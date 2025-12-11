import runpod


def handler(event):
    """
    Мінімальний handler для RunPod.
    Просто повертає те, що прийшло, плюс статус.
    """
    input_data = event.get("input", {}) or {}
    prompt = input_data.get("prompt", "")

    return {
        "ok": True,
        "mode": "echo",
        "received_prompt": prompt,
        "raw_event": event,
    }


# ВАЖЛИВО: саме ця функція запускає воркер і підключає його до черги RunPod.
runpod.serverless.start({"handler": handler})
