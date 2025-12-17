# 🚀 Crypto Market IA — Observatorio + API + Ingesta + Sentimiento + Predicción

> Plataforma ligera para capturar y normalizar datos cripto, enriquecer con sentimiento de noticias, exponer una API, visualizar en un panel y generar predicciones de corto plazo. _No es consejo financiero._

---

## 🎯 Objetivo del proyecto
- Unificar datos de mercado de fuentes públicas (CoinGecko/CoinPaprika).
- Mantener una serie temporal confiable (Parquet) para análisis y aprendizaje automático.
- Enriquecer el modelo con sentimiento diario a partir de titulares (CoinDesk).
- Publicar un observatorio visual (Streamlit) y una API (FastAPI).
- Automatizar ingesta y predicción periódica con Docker Compose.

---

## 🧭 Flujo general

```
CoinGecko/CoinPaprika  ──>  ingest-loop  ──>  data/market_snapshots.parquet
CoinDesk (titulares)   ──>  sentiment-loop  ──>  data/sentiment_coindesk.csv
Parquet + Sentimiento  ──>  predict-loop  ──>  data/export/models/*
API (FastAPI)          ──>  Panel (Streamlit)
```

### 1) Ingesta de mercado
- Fuentes: CoinGecko + CoinPaprika.
- Salida: `data/market_snapshots.parquet`.
- Controlada por `INGEST_INTERVAL_SECONDS`.

### 2) Sentimiento de mercado
- Extracción de titulares CoinDesk + análisis VADER.
- Salida: `data/sentiment_coindesk.csv`.
- Controlada por `SENTIMENT_INTERVAL_SECONDS`.

### 3) Predicción
- Modelo Conv1D + LSTM por símbolo.
- Variables de entrada:
  - `log_return` del precio.
  - `daily_sentiment` (promedio diario de titulares; 0 si no hay noticias).
- Salidas:
  - `data/export/models/<SYM>_model.keras`
  - `data/export/models/<SYM>_meta.json`
  - `data/export/models/latest_prediction_<SYM>.json`
  - `data/export/models/prediction_history_<SYM>.csv`
- Frecuencia: `PREDICT_INTERVAL_SECONDS`.
- Horizonte temporal: `PREDICT_STEP_MINUTES`.

---

## 🛠️ Requisitos

**Opción A (recomendada): Docker**
- Docker + Docker Compose.

**Opción B (local): Python**
- Python 3.11
- `pip install -r requirements.txt`

---

## ⚡ Inicio rápido (Docker)

1) Configura variables de entorno:
```bash
cp .env.example .env
```
Recomendado para horizonte de 5 minutos: `INGEST_INTERVAL_SECONDS=300` y `PREDICT_STEP_MINUTES=5`.

2) Levanta todos los servicios:
```bash
docker compose up --build -d api dashboard ingest-loop sentiment-loop predict-loop
```

3) Verifica servicios:
```bash
curl http://localhost:8000/health
```

4) Accesos:
- Documentación de la API: http://localhost:8000/docs
- Panel: http://localhost:8501

---

## ⏱️ Tiempos de espera (estimados)
- **Construcción inicial (Docker)**: 5–15 min (TensorFlow es pesado).
- **Primeros datos de mercado**: 1–2 min tras iniciar `ingest-loop`.
- **Primeros datos de sentimiento**: 1–2 min tras iniciar `sentiment-loop`.
- **Primeras predicciones**:
  - Ventana 12 y paso 5 min: ~60 min de datos.
  - Ventana reducida automática (mín. 4): ~20–30 min.

Si no hay predicciones aún, revisa logs con `docker compose logs -f predict-loop`.

---

## 🧩 Comandos útiles

### Docker (recomendado)
- Ingesta manual (mercado): `docker compose run --rm ingest`
- Sentimiento manual: `docker compose run --rm sentiment`
- Registros: `docker compose logs -f`
- Detener servicios: `docker compose down` (no uses `-v` si quieres conservar datos)

### Makefile (local)
- `make ingest`
- `make sentiment`
- `make api`
- `make dashboard`

---

## ⚙️ Configuración del .env (significado)

### API y red
- `API_HOST`: host del API.
- `API_PORT`: puerto del API.
- `API_BASE_URL`: URL base que usa el panel para consumir la API.

### Ingesta de mercado
- `REQUEST_TIMEOUT_SECONDS`: timeout HTTP por request.
- `TOP_N_ASSETS`: top N por capitalización de mercado.
- `INGEST_INTERVAL_SECONDS`: cada cuantos segundos se ingesta.

### Datos y almacenamiento
- `DATA_DIR`: carpeta de datos (en Docker se usa el volumen `market_data`).
- `PARQUET_FILENAME`: archivo Parquet principal.
- `SYMBOL_ALLOWLIST`: filtro global por símbolos (afecta API, panel y modelo).

### Sentimiento
- `SENTIMENT_INTERVAL_SECONDS`: cada cuanto se refresca el sentimiento.
- `SENTIMENT_HEADLINES_LIMIT`: cantidad de titulares por corrida.

### Predicción
- `PREDICT_SYMBOLS`: símbolos a entrenar/predicir (idealmente subset del allowlist).
- `PREDICT_INTERVAL_SECONDS`: cada cuánto se entrena/predice.
- `PREDICT_STEP_MINUTES`: cuántos minutos representa cada paso del pronóstico (T+1).
- `PREDICT_EPOCHS`: epochs de entrenamiento.
- `PREDICT_BLEND_ALPHA`: mezcla entre modelo y baseline (0-1). Menor = más conservador.
- `PREDICT_RETURN_CLIP_MULT`: multiplicador de volatilidad para limitar retornos.
- `PREDICT_MIN_RETURN_CLIP`: límite mínimo absoluto del retorno predicho.

**Nota**: `PREDICT_INTERVAL_SECONDS` no tiene que ser igual a `PREDICT_STEP_MINUTES`.  
El primero controla la frecuencia de entrenamiento; el segundo etiqueta el horizonte temporal.

---

## 📦 Salidas y artefactos

- `data/market_snapshots.parquet`: serie temporal de mercado.
- `data/sentiment_coindesk.csv`: sentimiento diario.
- `data/export/models/`: modelos y predicciones por símbolo.

---

## 🛡️ Solución de problemas (rápido)

- **No hay predicciones**: aún no hay suficientes datos. Espera más tiempo o reduce `window`.
- **Panel en blanco**: revisa `SYMBOL_ALLOWLIST` y que `/markets` tenga datos.
- **CPU alta**: baja `PREDICT_EPOCHS` o aumenta `PREDICT_INTERVAL_SECONDS`.
- **Predicciones muy agresivas**: baja `PREDICT_BLEND_ALPHA` y/o reduce `PREDICT_RETURN_CLIP_MULT`.
- **Errores de red**: revisa límites de tasa de CoinGecko/CoinPaprika.

---

## Uso responsable

Este proyecto es educativo. No recomienda inversiones ni garantiza exactitud en predicciones.

---

## 👥 Integrantes
- Juan Pablo Pérez
- Julián Ruiz
- Gabrial Imbacuán
