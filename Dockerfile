FROM python:3.13-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /auto-slide-tracker

RUN mkdir -p logs media && chmod a+rw logs media

COPY app ./app
COPY logs/logging_config.json logs/
COPY logs/logging_config.py logs/
COPY requirements.txt .

RUN pip install --upgrade pip && pip install -r requirements.txt

EXPOSE 8000

CMD ["hypercorn", "app.main:app", "--bind", "0.0.0.0:8000"]