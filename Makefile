# ChronoForge · 常用命令（作者：晨星）
PYTHON ?= python
PYTEST ?= $(PYTHON) -m pytest

.PHONY: help install dev test cov lint fmt format demo detcheck clean build docker

help:
	@echo "ChronoForge targets:"
	@echo "  make install   安装运行时依赖"
	@echo "  make dev       安装开发依赖（pytest/pytest-cov/ruff）"
	@echo "  make test      跑测试"
	@echo "  make cov       测试 + 覆盖率（门槛 80%）"
	@echo "  make lint      ruff check"
	@echo "  make fmt       ruff format"
	@echo "  make demo      端到端基准 + benchmark.json"
	@echo "  make detcheck  确定性校验"
	@echo "  make docker    构建镜像"

install:
	$(PYTHON) -m pip install -r requirements.txt

dev: install
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTEST) -q

cov:
	$(PYTEST) -q --cov=chronoforge --cov-report=term-missing --cov-fail-under=80

lint:
	$(PYTHON) -m ruff check .

fmt:
	$(PYTHON) -m ruff format .

demo:
	$(PYTHON) -m chronoforge.cli demo --out benchmark.json

detcheck:
	$(PYTHON) -m chronoforge.cli detcheck

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov build dist *.egg-info
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +

build: clean
	$(PYTHON) -m pip install -q build
	$(PYTHON) -m build

docker:
	docker build -t chronoforge:0.1.0 .
