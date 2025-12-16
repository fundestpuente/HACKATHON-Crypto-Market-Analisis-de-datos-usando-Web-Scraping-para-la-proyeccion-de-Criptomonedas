from __future__ import annotations

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from pathlib import Path
import json

from app.config import settings

API_BASE = settings.resolved_api_base_url.rstrip("/")
LOCAL_TZ = "America/Guayaquil"

st.set_page_config(page_title="Observatorio Cripto con IA", layout="wide")
st.title("🚀 Observatorio Cripto con IA")
st.caption("Datos unificados del mercado + dataset listo para TensorFlow + panel interactivo")
st.markdown(
    """
    <style>
    body, input, select, textarea {font-family: 'Inter', 'Helvetica', sans-serif !important;}
    h1, h2, h3, h4 {font-weight: 700;}
    .stMetric {background: #0b1221; border-radius: 12px; padding: 12px;}
    [data-testid="stSidebar"] {background: #0f172a; color: #e2e8f0;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=60)
def load_data() -> pd.DataFrame:
    try:
        resp = requests.get(f"{API_BASE}/markets", params={"limit": 500}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        df = pd.DataFrame(data)
        if df.empty:
            return df
        df["as_of"] = pd.to_datetime(df["as_of"], utc=True)
        df["as_of_local"] = df["as_of"].dt.tz_convert(LOCAL_TZ)
        return df
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not fetch data from the API: {exc}")
        return pd.DataFrame()


def render_prediction_section(filtered: pd.DataFrame) -> None:
    """Hero section for latest prediction + history."""
    st.markdown(
        """
        <style>
        .pred-card {
            background: linear-gradient(135deg, #0ea5e9 0%, #111827 80%);
            border-radius: 16px;
            padding: 18px 20px;
            color: #e2e8f0;
            margin-bottom: 14px;
            box-shadow: 0 12px 24px rgba(0,0,0,0.35);
        }
        .pred-grid {display: flex; gap: 24px; flex-wrap: wrap;}
        .pred-metric {min-width: 160px;}
        .pred-metric .label {font-size: 0.85rem; opacity: 0.85;}
        .pred-metric .value {font-size: 1.4rem; font-weight: 700;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("## 🔮 Predicción destacada")
    prediction_payload = None
    prediction_symbol = None
    prediction_files = list(Path(settings.data_dir / Path("export/models")).glob("latest_prediction_*.json"))
    available_pred_syms = sorted({p.stem.replace("latest_prediction_", "") for p in prediction_files})
    if available_pred_syms:
        if settings.symbol_allowlist:
            allowed_pred = [s for s in available_pred_syms if s.upper() in {x.upper() for x in settings.symbol_allowlist}]
        else:
            allowed_pred = available_pred_syms
        available_pred_syms = allowed_pred
        prediction_symbol = st.selectbox("Símbolo para predicción", options=available_pred_syms, index=0)
    else:
        st.info("Aún no hay archivos de predicción. Ejecuta predict-loop o train_and_predict.")
    if prediction_symbol:
        pred_file = Path(settings.data_dir) / "export" / "models" / f"latest_prediction_{prediction_symbol}.json"
        if pred_file.exists():
            try:
                prediction_payload = json.loads(pred_file.read_text())
            except Exception:
                prediction_payload = None

    if prediction_payload and prediction_payload.get("predicted_prices"):
        sym = prediction_payload.get("symbol", prediction_symbol or "N/A")
        last_price = prediction_payload["last_price"]
        pred_prices = prediction_payload["predicted_prices"]
        horizon = prediction_payload.get("horizon", len(pred_prices))
        forecast_times_raw = prediction_payload.get("forecast_times") or []
        forecast_times = pd.to_datetime(forecast_times_raw, utc=True)
        predicted_at_raw = prediction_payload.get("predicted_at")
        predicted_at = pd.to_datetime(predicted_at_raw, utc=True) if predicted_at_raw else None
        predicted_at_local = predicted_at.tz_convert(LOCAL_TZ) if predicted_at is not None else None
        forecast_times_local = forecast_times.tz_convert(LOCAL_TZ)

        st.markdown(
            f"""
            <div class="pred-card">
                <div style="font-size:0.95rem; opacity:0.9;">Última corrida</div>
                <div style="font-size:1.4rem; font-weight:700; margin-bottom:10px;">{sym}</div>
                <div class="pred-grid">
                    <div class="pred-metric">
                        <div class="label">Generado</div>
                        <div class="value">{predicted_at_local.strftime("%Y-%m-%d %H:%M %Z") if predicted_at_local is not None else "N/D"}</div>
                    </div>
                    <div class="pred-metric">
                        <div class="label">Precio actual</div>
                        <div class="value">{last_price:,.4f}</div>
                    </div>
                    <div class="pred-metric">
                        <div class="label">Próximo paso</div>
                        <div class="value">{pred_prices[0]:,.4f}</div>
                    </div>
                    <div class="pred-metric">
                        <div class="label">Horizonte</div>
                        <div class="value">{horizon} paso(s)</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Historical tail for context (full series, smoothed)
        tail = filtered[filtered["symbol"] == sym].sort_values("as_of").copy()
        tail = tail[["as_of_local", "price"]].dropna().sort_values("as_of_local")
        if not tail.empty:
            tail = (
                tail.set_index("as_of_local")
                .resample("30s")
                .median()
                .interpolate(method="linear")
                .reset_index()
            )
        hist_df = pd.DataFrame({"as_of": tail["as_of_local"], "price": tail["price"], "label": "Actual"})

        history_file = Path(settings.data_dir) / "export" / "models" / f"prediction_history_{sym}.csv"
        hist_pred_df = pd.DataFrame()
        if history_file.exists():
            try:
                hist_pred_df = pd.read_csv(history_file, parse_dates=["predicted_at", "forecast_time"])
                hist_pred_df["predicted_at"] = pd.to_datetime(hist_pred_df["predicted_at"], utc=True)
                hist_pred_df["forecast_time"] = pd.to_datetime(hist_pred_df["forecast_time"], utc=True)
                hist_pred_df = hist_pred_df.assign(label="Prediction history")
                hist_pred_df = hist_pred_df.rename(columns={"forecast_time": "as_of", "predicted_price": "price"})
                hist_pred_df["as_of"] = hist_pred_df["as_of"].dt.tz_convert(LOCAL_TZ)
                hist_pred_df["predicted_at"] = hist_pred_df["predicted_at"].dt.tz_convert(LOCAL_TZ)
            except Exception:
                hist_pred_df = pd.DataFrame()

        pred_df = pd.DataFrame(
            {
                "as_of": forecast_times_local,
                "price": pred_prices,
                "label": "Predicción (última)",
            }
        ).dropna(subset=["as_of", "price"])

        plot_df = pd.concat([hist_df, hist_pred_df, pred_df], ignore_index=True).sort_values("as_of")
        if plot_df.empty:
            st.info("No hay datos para graficar predicciones.")
        else:
            fig_pred = px.line(
                plot_df,
                x="as_of",
                y="price",
                color="label",
                title=f"{sym} | Serie actual vs predicciones publicadas",
                markers=True,
                color_discrete_map={
                    "Actual": "#0ea5e9",
                    "Prediction history": "#ef4444",
                    "Predicción (última)": "#f59e0b",
                },
            )
            fig_pred.update_traces(line_shape="linear", line=dict(width=2), marker=dict(size=5, opacity=0.65))
            fig_pred.update_layout(hovermode="x unified", legend_title_text="")
            st.plotly_chart(fig_pred, use_container_width=True)
    else:
        st.warning("Aún no hay predicciones disponibles. Asegúrate de que predict-loop o train_and_predict estén corriendo.")

df = load_data()

if df.empty:
    st.warning("No data available yet. Run the ingestion pipeline first.")
    st.stop()

sources = sorted(df["source"].unique())
symbols = sorted(df["symbol"].unique())

with st.sidebar:
    st.subheader("Filtros")
    selected_sources = st.multiselect("Fuentes", options=sources, default=sources)
    allowed = {s.upper() for s in settings.symbol_allowlist} if settings.symbol_allowlist else set(symbols)
    filtered_symbols = [s for s in symbols if s.upper() in allowed]
    selected_symbols = st.multiselect("Símbolos", options=filtered_symbols, default=filtered_symbols[: min(5, len(filtered_symbols))])
    if st.button("Refrescar datos"):
        st.cache_data.clear()
        st.rerun()

filtered = df[df["source"].isin(selected_sources) & df["symbol"].isin(selected_symbols)]
latest_prices = filtered.sort_values("as_of").groupby("symbol")["price"].last().to_dict()
latest_capture_local = filtered["as_of_local"].max()
latest_capture_str = latest_capture_local.strftime("%Y-%m-%d %H:%M %Z") if pd.notna(latest_capture_local) else "N/A"
cols = st.columns(3)
cols[0].metric(
    "Última captura",
    value=latest_capture_str,
    help="Marca de tiempo de la ingesta más reciente.",
)
if selected_symbols:
    sym0 = selected_symbols[0]
    cols[1].metric(f"{sym0} precio actual", f"{latest_prices.get(sym0, 0):,.2f}")
if len(selected_symbols) > 1:
    sym1 = selected_symbols[1]
    cols[2].metric(f"{sym1} precio actual", f"{latest_prices.get(sym1, 0):,.2f}")

# Predicción destacada primero
render_prediction_section(filtered)
st.divider()

col1, col2 = st.columns([1.2, 1])
with col1:
    top_market_cap = (
        filtered.groupby("name", as_index=False)["market_cap"]
        .mean(numeric_only=True)
        .nlargest(15, "market_cap")
    )
    fig = px.bar(
        top_market_cap,
        x="market_cap",
        y="name",
        orientation="h",
        color="market_cap",
        color_continuous_scale="Viridis",
        title="Top activos por capitalización (promedio entre fuentes)",
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    change = (
        filtered.groupby(["symbol", "source"], as_index=False)["change_24h"]
        .mean(numeric_only=True)
        .pivot(index="symbol", columns="source", values="change_24h")
    )
    heatmap = px.imshow(
        change,
        color_continuous_scale="RdYlGn",
        origin="lower",
        aspect="auto",
        labels=dict(color="Variación 24h %"),
        title="Mapa de calor: variación 24h",
    )
    st.plotly_chart(heatmap, use_container_width=True)

with st.container():
    st.subheader("Evolución de precios (símbolos seleccionados)")
    latest_symbols = filtered[filtered["symbol"].isin(selected_symbols[:5])].copy()
    fig = px.line(
        latest_symbols.sort_values("as_of_local"),
        x="as_of_local",
        y="price",
        color="symbol",
        facet_col="source",
        facet_col_wrap=2,
        title="Precio en el tiempo por fuente (hora America/Guayaquil)",
    )
    fig.update_xaxes(title_text="Fecha/hora")
    fig.update_yaxes(title_text="Precio")
    st.plotly_chart(fig, use_container_width=True)
