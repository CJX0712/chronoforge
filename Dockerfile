# syntax=docker/dockerfile:1
# ChronoForge · 作者：晨星
FROM python:3.12-slim

LABEL org.opencontainers.image.title="chronoforge"
LABEL org.opencontainers.image.description="ChronoForge 时间序列预测系统 · ChronoFuse 融合旗舰"
LABEL org.opencontainers.image.authors="晨星"
LABEL org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md ./
COPY chronoforge ./chronoforge
COPY tests ./tests
COPY examples ./examples

# 依赖可选：statsmodels 缺失时自动降级为纯 numpy 基池，离线可跑。
RUN python -m pip install --no-cache-dir -e . || true

# 冒烟验证：端到端 demo 必须能跑通
RUN python -m chronoforge.cli demo --out /tmp/benchmark.json

ENTRYPOINT ["python", "-m", "chronoforge.cli"]
CMD ["demo"]
