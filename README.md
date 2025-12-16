# Crypto Market AI — Observatorio + API + Ingesta + Predicción
[Leer en español](README.es.md)

> Stack ligera para **capturar y normalizar datos cripto**, exponerlos vía **API**, visualizarlos en un **dashboard** y generar **predicciones automáticas**. Transparencia y trazabilidad primero. _No es consejo financiero._

---

## 1) Planteamiento del problema (resumen)
La información pública de mercado cripto suele estar **fragmentada**, en **formatos heterogéneos** y con poca **trazabilidad**. Esto hace lento y propenso a errores el análisis básico, especialmente para usuarios nuevos. Proponemos una pila simple que automatiza ingesta (APIs públicas), **estandariza** campos y ofrece **API + dashboard + predicciones** para comparar y monitorear rápidamente.

---

## 2) Objetivos del proyecto
- **Automatizar** la recolección periódica de datos públicos (precio, cambio 24h, volumen 24h, market cap, timestamp, fuente).
- **Estandarizar** el modelo de datos para comparar múltiples fuentes por activo.
- **Exponer** resultados vía **REST API** y **visualizarlos** en un **observatorio** (Streamlit).
- **Agregar predicciones** automáticas (entrenamiento/predicción periódicos) y mostrar historial de predicciones en el dashboard.
- **Trazabilidad y transparencia**: mostrar fuente, momento de captura, método y limitaciones.
- **Sin dark patterns**: no se venden cursos ni afiliados ni se incentiva la compra.

---

## 3) Herramientas utilizadas
- **Backend (API):** Python + FastAPI.
- **Ingesta:** httpx + Pandas; fuentes públicas CoinGecko/CoinPaprika; almacenamiento Parquet.
- **Dashboard:** Streamlit + Plotly; Requests para consumir la API; selección dinámica de símbolos (SYMBOL_ALLOWLIST).
- **Predicciones:** TensorFlow (Conv1D + LSTM), bucle de entrenamiento/predicción automático (predict-loop).
- **Schedulers:** ingest-loop (ingesta periódica) y predict-loop (entrena/predice en intervalos).
- **Orquestación:** Docker Compose (volumen `market_data` para persistir datos/predicciones).

---

## 4) Instalación y puesta en marcha (Docker)

### 4.1 Prerrequisitos
- Docker + Docker Compose.

### 4.2 Preparar entorno
```bash
cd NEW
cp .env.example .env   # ajusta SYMBOL_ALLOWLIST, intervalos, puertos si es necesario
```

### 4.3 Levantar todos los servicios
```bash
docker compose up --build -d api dashboard ingest-loop predict-loop
```
- **ingest-loop**: ingesta cada `INGEST_INTERVAL_SECONDS` (default 300s).
- **predict-loop**: entrena/predice para `PREDICT_SYMBOLS` cada `PREDICT_INTERVAL_SECONDS` (default 600s); guarda modelos y predicciones en `data/export/models/`.

### 4.4 Comandos útiles
- Ingesta manual: `docker compose run --rm ingest`
- Exportar CSV: `docker compose run --rm api python -m scripts.export_dataset --output data/export/market_dataset.csv`
- Copiar CSV al host: `docker compose run --rm api sh -c "cat data/export/market_dataset.csv" > market_dataset.csv`
- Entrenar/predicir BTC y DOGE (una vez): `docker compose run --rm api python -m scripts.train_and_predict --symbols BTC,DOGE --window 12 --horizon 1 --source CoinPaprika`

### 4.5 Accesos rápidos
- API docs: http://localhost:8000/docs
- Dashboard: http://localhost:8501
- Datos filtrados por `SYMBOL_ALLOWLIST` en todo el pipeline (API, dashboard, entrenamiento/predicción).
- Predicciones/historial: `data/export/models/` (`latest_prediction_<SYM>.json`, `prediction_history_<SYM>.csv`).

### 4.6 Salud y revisiones rápidas
- API health: `http://localhost:8000/health`
- Ver logs: `docker compose logs -f`
- Conservar datos/predicciones: no uses `docker compose down -v` (el volumen `market_data` guarda Parquet y predicciones).

### 4.7 Colab (opcional)
- Exporta `market_dataset.csv` y súbelo, o usa `API_BASE_URL` vía ngrok. Ajusta `WINDOW_SIZE` pequeño si hay pocos puntos; el notebook se adapta a datos escasos.

---

## 5) Qué entrega este prototipo
Un **servicio FastAPI**, bucles de **ingesta** y **predicción**, y un **dashboard Streamlit** que:
- Automatiza la captura y normalización de snapshots de mercado.
- Expone `/markets` filtrado por allowlist, y muestra métricas y gráficos.
- Entrena y publica predicciones periódicas; el dashboard muestra historial de predicciones para contrastar con lo realizado.
- Es extensible: puedes ampliar allowlist, fuentes o ajustar intervalos sin cambiar la arquitectura.

---

## 6) Cómo funciona el modelo de predicción (IA)
- **Datos y features:** cada símbolo se filtra por `SYMBOL_ALLOWLIST`, se ordena por `as_of` y se calcula el **log-return** (`log(price).diff()`) para trabajar con una serie más estacionaria.
- **Ventanas y horizonte:** se generan ventanas deslizantes de longitud `--window` (por defecto 12 en `train_and_predict.py`, 72 en `train_model.py`) para predecir `--horizon` pasos futuros de log-return. Si los datos son escasos, la ventana se reduce automáticamente para no quedarse sin ejemplos.
- **Arquitectura (Conv1D + LSTM apiladas):** entrada `(window, 1)` → `Conv1D(32, k=3, padding="causal", relu)` → `SpatialDropout1D(0.1)` → `LSTM(64, return_sequences=True)` → `Dropout(0.15)` → `LSTM(32)` → `Dense(16, relu)` → `Dense(horizon)` (salida lineal de log-returns). Regularización L2 suave en capas densas/recurrentes.
- **Entrenamiento:** normaliza con media/desv. estándar del set de entrenamiento; split aproximado 70/15/15 (train/val/test, con salvaguarda para muestras pequeñas). Optimizer Adam (`lr=1e-3`, `clipnorm=1.0`), `loss="mse"`, `metrics=["mae"]`, callbacks de `EarlyStopping` (paciencia 8, `min_delta=1e-5`) y `ReduceLROnPlateau` (factor 0.5, `min_lr=1e-5`).
- **Salidas y artefactos:** por símbolo se guarda `data/export/models/<SYM>_model.keras` y `<SYM>_meta.json` (ventana, horizonte, media/desv., historial de pérdida, métricas de test si existen). Cada corrida escribe `latest_prediction_<SYM>.json` (retornos y precios pronosticados con timestamps) y acumula historial en `prediction_history_<SYM>.csv`.
- **Ejecución:** manual con `python -m scripts.train_and_predict --symbols BTC,DOGE --window 12 --horizon 1 --source CoinPaprika` o en bucle con `predict-loop` (Docker Compose) que reentrena y publica predicciones cada `PREDICT_INTERVAL_SECONDS`.

---

### Estructura del repositorio
```
app/
  api/           # FastAPI app y rutas
  dashboard/     # Streamlit UI (observatorio + predicciones)
  ingestion/     # Clientes de fuentes y pipeline
  utils/         # Config/logging helpers
  storage.py     # Parquet store y exportador CSV
  config.py      # Settings (env-driven)
scripts/         # ingest_once, export_dataset, train_model, train_and_predict, loops
data/            # Parquet/exports (persistido en volumen market_data)
docker-compose.yml
Makefile
README.md / README.es.md
```

### Nota de uso responsable
Este proyecto **no recomienda inversión** ni promueve compras de criptoactivos. Es **educativo** y orientado a **alfabetización de datos**: ofrece información pública trazable para que cada usuario la evalúe por su cuenta.
