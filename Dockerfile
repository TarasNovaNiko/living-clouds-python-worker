# living-clouds-python/Dockerfile

# Базовий образ з Python
FROM python:3.11-slim

# Щоб Python не буферизував лог і одразу писав у stdout
ENV PYTHONUNBUFFERED=1

# Вмикаємо використання Stable Diffusion всередині контейнера
# (локально ми залишимо USE_SD=0, а в контейнері буде USE_SD=1)
ENV USE_SD=1

# Створюємо робочу директорію всередині контейнера
WORKDIR /app

# Копіюємо файли проєкту (тільки бекенд) у контейнер
COPY . /app

# Встановлюємо залежності
RUN pip install --no-cache-dir -r requirements.txt

# Відкриваємо порт 8000 всередині контейнера
EXPOSE 8000

# Команда запуску FastAPI через uvicorn
CMD ["python", "-u", "worker.py"]
