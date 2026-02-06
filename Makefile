.PHONY: install test lint format docker-build docker-run clean help

help:
	@echo "Available commands:"
	@echo "  make install       - Install package with dev dependencies"
	@echo "  make test          - Run tests with coverage"
	@echo "  make lint          - Run linters (flake8, mypy)"
	@echo "  make format        - Format code with black and isort"
	@echo "  make docker-build  - Build Docker image locally"
	@echo "  make docker-run    - Run Docker container"
	@echo "  make clean         - Clean build artifacts and caches"

install:
	pip install --upgrade pip setuptools wheel
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pip install -e .
	@echo "Installation complete! Run 'pre-commit install' to enable git hooks."

test:
	pytest tests/ -v --cov=src/robust_vision --cov-report=term --cov-report=html

lint:
	@echo "Running flake8..."
	flake8 src/ scripts/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 src/ scripts/ tests/ --count --max-complexity=10 --max-line-length=100 --statistics
	@echo "Running mypy..."
	mypy src/robust_vision --ignore-missing-imports || true

format:
	@echo "Formatting with black..."
	black src/ scripts/ tests/
	@echo "Sorting imports with isort..."
	isort --profile black src/ scripts/ tests/

docker-build:
	docker build -t robust-vision:local .

docker-run:
	docker run --rm -it --gpus all robust-vision:local

docker-compose-up:
	docker-compose up -d

docker-compose-down:
	docker-compose down

clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name htmlcov -exec rm -rf {} + 2>/dev/null || true
	rm -f .coverage coverage.xml
	@echo "Clean complete!"
