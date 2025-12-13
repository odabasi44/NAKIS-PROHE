FROM python:3.11-slim

# Uygulama klasörü
WORKDIR /app

# OpenCV için gereken bazı sistem kütüphaneleri
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Python bağımlılıkları
COPY requirements.txt .
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Tüm kodları kopyala
COPY . .

# Flask / gunicorn’un dinleyeceği port
EXPOSE 5001

# Uygulamayı başlat (Coolify/Platform PORT env veriyorsa onu kullan)
CMD ["sh", "-c", "gunicorn -b 0.0.0.0:${PORT:-5001} run:app"]
