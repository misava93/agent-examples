.PHONY: venv-create
venv-create:
	@echo "Creating virtual environment"
	python -m venv .venv

.PHONY: install-deps
install-deps:
	@echo "Installing dependencies"
	pip install -r requirements.txt

.PHONY: lint-check
lint-check:
	@echo "Running lint check"
	black --check .

.PHONY: fmt
lint-fmt:
	@echo "Running lint formatting"
	black .

.PHONY: test
test:
	@echo "Running tests"
	pytest -v .

.PHONY: start-dev
api-start-dev:
	@echo "Starting API in development mode"
	fastapi dev main.py

.PHONY: start-prod
api-start-prod:
	@echo "Starting API in production mode"
	fastapi run main.py

.PHONY: test-api
test-api:
	@echo "Testing API"
	curl http://127.0.0.1:8000/health -H "Content-Type: application/json"

.PHONY: run-bug-fixer-agent
run-bug-fixer-agent:
	@echo "Running bug fixer agent"
	python bug_fixer_agent.py
