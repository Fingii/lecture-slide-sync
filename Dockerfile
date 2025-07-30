FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /auto-slide-tracker

RUN mkdir -p logs && chmod a+rw logs

COPY . .

RUN pip install --upgrade pip && pip install -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]