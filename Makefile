# Makefile pour le projet Robot Framework Agent

.PHONY: help install install-dev test lint format sync-requirements clean

help:  ## Afficher cette aide
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Installer en mode production
	pip install .

install-dev:  ## Installer en mode développement
	pip install -e ".[dev]"
	pre-commit install

test:  ## Lancer les tests
	pytest tests/utest/ -v

test-cov:  ## Lancer les tests avec coverage
	pytest tests/utest/ --cov=Agent --cov-report=html --cov-report=term

lint:  ## Vérifier le code (flake8 + mypy)
	flake8 Agent/ tests/
	mypy Agent/ --ignore-missing-imports

format:  ## Formater le code (black + isort)
	black Agent/ tests/
	isort Agent/ tests/

format-check:  ## Vérifier le formatage sans modifier
	black --check Agent/ tests/
	isort --check Agent/ tests/

sync-requirements:  ## Synchroniser requirements.txt depuis pyproject.toml (nécessite pip-tools)
	@echo "⚠️  Vérifiez que pip-tools est installé: pip install pip-tools"
	pip-compile pyproject.toml -o requirements.txt --no-header
	@echo "# Robot Framework Agent - Development Dependencies" > requirements-dev.tmp
	@echo "# Install: pip install -r requirements-dev.txt" >> requirements-dev.tmp
	@echo "" >> requirements-dev.tmp
	@echo "-r requirements.txt  # Include production dependencies" >> requirements-dev.tmp
	@echo "" >> requirements-dev.tmp
	pip-compile --extra dev pyproject.toml --no-header | grep -v "robotframework\|openai\|Pillow\|tiktoken\|python-dotenv\|requests" >> requirements-dev.tmp
	mv requirements-dev.tmp requirements-dev.txt
	@echo "✅ requirements.txt et requirements-dev.txt synchronisés depuis pyproject.toml"

clean:  ## Nettoyer les fichiers générés
	rm -rf build/ dist/ *.egg-info
	rm -rf .pytest_cache/ .coverage htmlcov/
	rm -rf Agent/__pycache__/ Agent/**/__pycache__/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build:  ## Construire le package
	python -m build

publish-test:  ## Publier sur TestPyPI
	twine upload --repository testpypi dist/*

publish:  ## Publier sur PyPI
	twine upload dist/*

pre-commit:  ## Lancer pre-commit sur tous les fichiers
	pre-commit run --all-files

