.PHONY: install ingest api dashboard export sentiment lint fmt

install:
	pip install -r requirements.txt

ingest:
	python -m scripts.ingest_once

sentiment:
	python -m scripts.sentiment_once

api:
	uvicorn app.api.main:app --reload --host 0.0.0.0 --port $${API_PORT:-8000}

dashboard:
	streamlit run app/dashboard/dashboard_app.py

export:
	python -m scripts.export_dataset --output data/export/market_dataset.csv

# --- Dockerized workflow ---
compose-up:
	docker compose up --build -d api dashboard ingest-loop sentiment-loop predict-loop

compose-ingest:
	docker compose run --rm ingest

compose-ingest-loop:
	docker compose up -d ingest-loop

compose-sentiment:
	docker compose run --rm sentiment

compose-sentiment-loop:
	docker compose up -d sentiment-loop

compose-export:
	docker compose run --rm api python -m scripts.export_dataset --output data/export/market_dataset.csv

compose-pull-csv:
	@docker compose run --rm api sh -c "cat data/export/market_dataset.csv" > market_dataset.csv
	@echo "CSV copied to ./market_dataset.csv"

compose-logs:
	docker compose logs -f

compose-down:
	docker compose down -v

compose-train:
	docker compose run --rm api python -m scripts.train_model --symbol BTC --window 72 --horizon 1

compose-predict:
	docker compose run --rm api python -m scripts.predict_next --symbol BTC --model-path data/export/models/BTC_model.keras --meta-path data/export/models/BTC_meta.json

compose-train-predict:
	docker compose run --rm api python -m scripts.train_and_predict --symbols BTC,DOGE --window 12 --horizon 1 --source CoinPaprika

compose-predict-loop:
	docker compose up -d predict-loop
