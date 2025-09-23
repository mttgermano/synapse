run:
	uv run --with streamlit streamlit run app.py

setup:
	pyenv virtualenv 3.10.12 venv && pyenv activate venv && uv pip install -r requirements.txt

install:
	uv pip install -r requirements.txt

docker-build:
	docker build -t synapse .

docker-run:
	docker run -it --rm -p 8501:8501 synapse
