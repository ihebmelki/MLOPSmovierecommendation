import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from fastapi import APIRouter
from starlette.responses import HTMLResponse
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64

router = APIRouter()

request_history = []

def add_request_to_history(user_id: int, n_recommendations: int):
    request_history.append({"user_id": user_id, "n_recommendations": n_recommendations})
    if len(request_history) > 1000:
        request_history.pop(0)

@router.get("/monitoring/drift-report", response_class=HTMLResponse)
def drift_report():
    if len(request_history) < 50:
        return f"<h1>Need 50+ requests (current: {len(request_history)})</h1><p>Make some /recommend calls first!</p>"

    # Split: ref = premières 200, current = dernières 200
    ref = pd.DataFrame(request_history[:200])
    cur = pd.DataFrame(request_history[-200:])

    # Tests de drift
    ks_user_pvalue = ks_2samp(ref["user_id"], cur["user_id"]).pvalue
    ks_n_pvalue = ks_2samp(ref["n_recommendations"], cur["n_recommendations"]).pvalue
    dataset_drift = (ks_user_pvalue < 0.05) or (ks_n_pvalue < 0.05)

    # Graphiques Plotly
    fig = make_subplots(rows=1, cols=2, subplot_titles=("User ID Distribution", "N Recommendations"))

    # User ID hist
    fig.add_trace(go.Histogram(x=ref["user_id"], name="Reference", opacity=0.7, nbinsx=20), row=1, col=1)
    fig.add_trace(go.Histogram(x=cur["user_id"], name="Current", opacity=0.7, nbinsx=20), row=1, col=1)

    # N rec hist
    fig.add_trace(go.Histogram(x=ref["n_recommendations"], name="Reference", opacity=0.7, nbinsx=10), row=1, col=2)
    fig.add_trace(go.Histogram(x=cur["n_recommendations"], name="Current", opacity=0.7, nbinsx=10), row=1, col=2)

    fig.update_layout(title="Data Drift Detection Report", height=400)
    plot_html = fig.to_html(full_html=False)

    # Rapport HTML complet
    html = f"""
    <html>
    <head><title>Drift Report</title></head>
    <body>
        <h1>ML Monitoring: Data Drift Report</h1>
        <p><strong>Generated:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Total requests analyzed:</strong> {len(request_history)}</p>
        <p><strong>Dataset Drift Detected:</strong> {'<span style="color:red">YES</span>' if dataset_drift else '<span style="color:green">NO</span>'}</p>
        <h3>Feature Drift Details:</h3>
        <ul>
            <li>User ID (KS p-value: {ks_user_pvalue:.4f}): {'Drift!' if ks_user_pvalue < 0.05 else 'Stable'}</li>
            <li>N Recommendations (KS p-value: {ks_n_pvalue:.4f}): {'Drift!' if ks_n_pvalue < 0.05 else 'Stable'}</li>
        </ul>
        <div>{plot_html}</div>
        <p><em>Powered by SciPy + Plotly (custom MLOps monitoring)</em></p>
    </body>
    </html>
    """
    return html
