# chart-pattern-analyze

Este MVP te permite escribir un **ticker** y ver **dos gráficos** (Diario y Semanal) con una línea **SMA200 turquesa** y **círculos rojos** marcando eventos.
Detecta (heurístico):

- Head & Shoulders (y su inverso), incluida una confirmación simple por ruptura de neckline.
- Breakout por encima del **máximo de N barras** (por defecto 252).
- Cup & Handle (versión muy simplificada).

> **Aviso**: Es un MVP. Los patrones complejos (H&S y Cup) usan heurísticas; puede haber falsos positivos/negativos.

## Requisitos

- Python 3.10+
- Instalar dependencias:
```
pip install -r requirements.txt
```

## Ejecutar

```
streamlit run app.py
```
Luego abre el enlace local que te muestra Streamlit.

## Personalización rápida
- Cambia la profundidad/anchura de detección en `detect_head_shoulders` y `detect_cup_handle`.
- Ajusta `lookback_breakout` desde la UI.
- Para exportar imágenes automáticamente, sustituye `st.pyplot` por guardado `fig.savefig(...)`.

## Próximos pasos
- Añadir endpoint HTTP (`/scan`, `/chart`) y convertir esto en **API**.
- Conectar un **Agent** (n8n/OpenAI tools) para que invoque la API y devuelva lista + PNGs.
- Añadir más patrones (MACD cross, Bollinger squeeze, etc.) y un **score** por ticker.
