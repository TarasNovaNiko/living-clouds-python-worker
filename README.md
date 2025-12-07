# Living Clouds Python Worker

Lightweight FastAPI backend for Living Clouds project.

- `/health` — healthcheck
- `/txt2img` — text-to-image (returns base64, currently mock/local)
- `/img2img` — image-to-image (returns base64, currently mock)

This project is intended to be deployed as a custom worker on RunPod
using the provided Dockerfile.
