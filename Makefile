.PHONY: venv-create
venv-create:
	@echo "Creating virtual environment"
	python -m venv .venv

.PHONY: venv-activate-fish
venv-activate-fish:
	@echo "Activating virtual environment"
	source .venv/bin/activate.fish

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
	fastapi dev --port 8085 main.py

.PHONY: start-prod
api-start-prod:
	@echo "Starting API in production mode"
	fastapi run --port 8085 main.py

.PHONY: run-ticket-bug-fixer-agent
run-ticket-bug-fixer:
	@echo "Running ticket bug fixer"
	python agents/ticket-bug-fixer.py
