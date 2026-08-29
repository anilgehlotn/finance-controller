export PYTHONPATH := src

.PHONY: data eval app test lint

data:
	python -m recon.generate_data

eval:
	python -m recon.eval

app:
	streamlit run src/recon/app.py

test:
	pytest -q

lint:
	ruff check .
