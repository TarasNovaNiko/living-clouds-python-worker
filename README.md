POST /run
{
  "input": {
    "mode": "img2img",
    "prompt": "turn clouds into a cute dragon, anime style",
    "init_image_url": "https://i.ibb.co/....jpg",
    "strength": 0.6,
    "steps": 20,
    "guidance": 7.5,
    "width": 512,
    "height": 512
  }
}
{
  "ok": true,
  "image_base64": "...",
  "meta": {...}
}

A txt2img

mode: "txt2img"

prompt, negative_prompt?, steps, guidance, width, height

B img2img

mode: "img2img"

prompt, init_image_url (саме прямий i.ibb.co/...jpg)

strength, steps, guidance, width, height

І відповідь: ok, image_base64, meta.