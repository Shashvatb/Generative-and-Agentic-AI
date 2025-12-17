FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip

RUN apt-get update && apt-get install -y \
    graphviz \
    graphviz-dev \
    build-essential \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN uv add --no-cache-dir -r requirements.txt
#RUN pip install --no-cache-dir -r requirements.txt
COPY . .

EXPOSE 7860

CMD ["python", "main.py"]