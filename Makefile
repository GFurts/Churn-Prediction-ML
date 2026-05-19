train:
	python -m src.train_baseline

api:
	uvicorn src.api:app --reload

test:
	pytest -v

lint:
	ruff check .

lint-fix:
	ruff check . --fix