```make
PY ?= python
ACT := . .venv/bin/activate;

setup:
	@test -d .venv || $(PY) -m venv .venv
	@$(ACT) pip install --upgrade pip && pip install -r requirements.txt

run:
	@$(ACT) streamlit run src/app_streamlit.py

clean:
	rm -rf __pycache__ results .pytest_cache *.pyc
