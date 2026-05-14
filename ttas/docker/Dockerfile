FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    libgl1-mesa-glx \
    rustc \
    cargo \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# For multipers, it might require compilation. If it fails, the python fallback is used in code.
# RUN pip install multipers || echo "multipers failed, falling back to giotto-tda/ripser"

COPY . /app

EXPOSE 8050

CMD ["python", "run_pipeline.py"]
