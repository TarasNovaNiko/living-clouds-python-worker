from fastapi import FastAPI
from pydantic import BaseModel
from model_stub import generate_txt2img, generate_img2img

app = FastAPI()


class Txt2ImgRequest(BaseModel):
    prompt: str


class Img2ImgRequest(BaseModel):
    prompt: str
    image_base64: str  # сюди пізніше будемо передавати фото хмар у base64


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/txt2img")
def txt2img(request: Txt2ImgRequest):
    """
    Викликаємо модель (поки що заглушку з model_stub.generate_txt2img).
    Пізніше всередині цієї функції буде справжній diffusers-пайплайн.
    """
    result = generate_txt2img(request.prompt)
    return result


@app.post("/img2img")
def img2img(request: Img2ImgRequest):
    """
    Викликаємо модель (поки що заглушку з model_stub.generate_img2img).
    Потім тут буде реальний StableDiffusionImg2ImgPipeline.
    """
    result = generate_img2img(request.prompt, request.image_base64)
    return result
