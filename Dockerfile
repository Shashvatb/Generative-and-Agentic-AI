FROM python:3.12-slim

WORKDIR /app

COPY . .
RUN pip install --upgrade pip

RUN apt-get update && apt-get install -y \
    graphviz \
    graphviz-dev \
    build-essential \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

RUN uv sync --frozen --no-cache
# RUN uv pip install --system

EXPOSE 7860

CMD ["uv", "run", "main.py"]