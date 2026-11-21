.PHONY: setup test train score app docker-up docker-down clean

setup:
	python -m venv .venv
	.venv/Scripts/python -m pip install --upgrade pip
	.venv/Scripts/python -m pip install -r requirements.txt

test:
	python -m compileall -q app src scripts

train:
	PYTHONPATH=src python src/train_baselines.py

score:
	PYTHONPATH=src python src/score_maintenance.py

app:
	python -m streamlit run app/streamlit_app.py --server.port=8502

docker-up:
	docker compose up --build

docker-down:
	docker compose down

clean:
	powershell -NoProfile -Command "Get-ChildItem -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force"

