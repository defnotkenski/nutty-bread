FROM nvidia/cuda:12.8.1-devel-ubuntu22.04
LABEL authors="kennylao"

RUN apt-get update
RUN apt-get install -y curl git

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /workspace

COPY pyproject.toml uv.lock ./
RUN uv sync

COPY . .
