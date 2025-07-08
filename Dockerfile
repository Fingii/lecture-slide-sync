FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY *.py .

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "fast_api_main:app", "--host", "0.0.0.0", "--port", "8000"]
