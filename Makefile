.PHONY: venv test lint format install-hooks

# 기본 Python 인터프리터
PYTHON := python3
VENV_DIR := venv
VENV_BIN := $(VENV_DIR)/bin
PIP_VENV := $(VENV_BIN)/pip

venv: ## 가상환경 생성 및 개발 의존성 설치
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "가상환경을 생성하는 중..."; \
		$(PYTHON) -m venv $(VENV_DIR); \
		echo "가상환경이 생성되었습니다."; \
	fi
	@echo "의존성을 설치하는 중..."
	$(PIP_VENV) install --upgrade pip
	$(PIP_VENV) install -r requirements-dev.txt
	@echo "설치 완료! 가상환경을 활성화하세요: source venv/bin/activate"

test: ## 테스트 실행
	$(VENV_BIN)/pytest tests/ -v

lint: ## 린트 실행 (오류 확인만)
	$(VENV_BIN)/ruff check src/

format: ## 포맷터 실행 (자동 수정)
	$(VENV_BIN)/ruff check --fix src/
	$(VENV_BIN)/ruff format src/

install-hooks: ## pre-commit hook 설치
	$(VENV_BIN)/pre-commit install
