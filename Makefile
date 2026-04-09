PYTHONPATH=.
PYTHON=python3
PIP=$(PYTHON) -m pip

.PHONY: install dataset ingest train api ui

install:
	$(PIP) install -r requirements.txt

dataset:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m train.generate_dataset

ingest:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m rag.ingest

train:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m train.train_lora --epochs 1

api:
	PYTHONPATH=$(PYTHONPATH) uvicorn app.backend.main:app --reload

ui:
	PYTHONPATH=$(PYTHONPATH) streamlit run app/frontend/streamlit_app.py
