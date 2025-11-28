install:
	@echo "Installing the application..."
	pip install --upgrade -r requirements.txt && pip install -r requirements.txt
	python -m pip install --upgrade pip

lint:
	@echo "Running linting checks..."
	# Add linting commands here
	pylint src/

test:
	@echo "Running tests..."
	python -m pytest tests/
