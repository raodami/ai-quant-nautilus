"""
Visualization module for backtest results.

Generates interactive charts using plotly and exports to HTML reports.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def equity_curve_plot(
    equity: list[float],
    benchmark: Optional[list[float]] = None,
    title: str = "Equity Curve",
    width: int = 1000,
    height: int = 500,
) -> str:
    """Generate HTML for equity curve chart."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        logger.warning("plotly not installed, skipping visualization")
        return ""

    dates = pd.date_range(start="2024-01-01", periods=len(equity), freq="D")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=equity,
        mode='lines', name='Strategy',
        line=dict(color='#6366f1', width=2),
    ))

    if benchmark is not None:
        fig.add_trace(go.Scatter(
            x=dates, y=benchmark,
            mode='lines', name='Benchmark',
            line=dict(color='#94a3b8', width=2, dash='dash'),
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        legend=dict(x=0.01, y=0.99),
        template="plotly_white",
        width=width,
        height=height,
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)


def drawdown_plot(
    drawdown: list[float],
    title: str = "Drawdown",
    width: int = 1000,
    height: int = 400,
) -> str:
    """Generate HTML for drawdown chart."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return ""

    dates = pd.date_range(start="2024-01-01", periods=len(drawdown), freq="D")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dates, y=drawdown,
        marker_color=[ '#ef4444' if v < -0.05 else '#f59e0b' if v < -0.02 else '#22c55e' for v in drawdown ],
        name="Drawdown",
    ))

    fig.add_hline(y=-0.10, line_dash="dash", line_color="red", annotation_text="-10%")
    fig.add_hline(y=-0.20, line_dash="dash", line_color="darkred", annotation_text="-20%")

    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        template="plotly_white",
        width=width,
        height=height,
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)


def trade_histogram(
    trades: list[dict],
    title: str = "Trade P&L Distribution",
    width: int = 800,
    height: int = 400,
) -> str:
    """Generate histogram of trade P&L."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return ""

    pnls = [t.get('pnl', 0) for t in trades]
    colors = ['#22c55e' if p > 0 else '#ef4444' for p in pnls]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=pnls,
        marker_color=colors,
        opacity=0.7,
        nbinsx=30,
        name="P&L",
    ))

    fig.add_vline(x=0, line_dash="dash", line_color="gray")

    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis_title="Profit/Loss ($)",
        yaxis_title="Count",
        template="plotly_white",
        width=width,
        height=height,
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)


def correlation_heatmap(
    corr_matrix: pd.DataFrame,
    title: str = "Asset Correlation Matrix",
    width: int = 600,
    height: int = 500,
) -> str:
    """Generate correlation heatmap."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return ""

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu_r',
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        hoverongaps=False,
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        template="plotly_white",
        width=width,
        height=height,
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)


def metrics_summary(result: dict) -> str:
    """Generate HTML summary of key metrics."""
    html = """
    <div style="font-family: sans-serif; padding: 20px;">
        <h2 style="color: #6366f1;">Backtest Summary</h2>
        <table style="border-collapse: collapse; width: 100%;">
    """
    
    metrics = [
        ("Total Return", result.get('total_return', 0), '%'),
        ("Annualized Return", result.get('annualized_return', 0), '%'),
        ("Sharpe Ratio", result.get('sharpe_ratio', 0), ''),
        ("Sortino Ratio", result.get('sortino_ratio', 0), ''),
        ("Max Drawdown", result.get('max_drawdown_pct', 0), '%'),
        ("Volatility", result.get('volatility', 0), '%'),
        ("Total Trades", result.get('total_trades', 0), ''),
        ("Win Rate", result.get('win_rate', 0), '%'),
        ("Profit Factor", result.get('profit_factor', 0), ''),
        ("Initial Capital", result.get('initial_capital', 0), '$'),
        ("Final Capital", result.get('final_capital', 0), '$'),
        ("Net PnL", result.get('net_pnl', 0), '$'),
    ]

    for name, value, unit in metrics:
        if unit == '%':
            formatted = f"{value * 100:.2f}"
        elif unit == '$':
            formatted = f"{value:,.2f}"
        else:
            formatted = f"{value:.4f}"
        
        html += f'<tr><td style="padding: 8px; border-bottom: 1px solid #e5e7eb;">{name}</td>'
        html += f'<td style="padding: 8px; border-bottom: 1px solid #e5e7eb; text-align: right; font-weight: bold;">{formatted}{unit}</td></tr>\n'

    html += "</table></div>"
    return html


def export_report(
    result: dict,
    output_path: str,
    equity: Optional[list[float]] = None,
    drawdown: Optional[list[float]] = None,
    trades: Optional[list[dict]] = None,
    correlation: Optional[pd.DataFrame] = None,
) -> Path:
    """Export complete backtest report to HTML file."""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Backtest Report - """ + result.get('strategy_name', 'Strategy') + """</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; padding: 20px; background: #f8fafc; }
            .container { max-width: 1200px; margin: 0 auto; }
            h1 { color: #6366f1; }
            .section { background: white; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Backtest Report</h1>
            <p>Strategy: """ + result.get('strategy_name', 'N/A') + """</p>
            <p>Generated: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
            
            <div class="section">
    """
    
    # Metrics summary
    html += metrics_summary(result)
    html += "</div>\n"

    # Equity curve
    if equity:
        html += '<div class="section"><h3>Equity Curve</h3>'
        html += equity_curve_plot(equity, title="Portfolio Equity")
        html += "</div>\n"

    # Drawdown
    if drawdown:
        html += '<div class="section"><h3>Drawdown</h3>'
        html += drawdown_plot(drawdown)
        html += "</div>\n"

    # Trade histogram
    if trades:
        html += '<div class="section"><h3>Trade Distribution</h3>'
        html += trade_histogram(trades)
        html += "</div>\n"

    # Correlation
    if correlation is not None:
        html += '<div class="section"><h3>Correlation Matrix</h3>'
        html += correlation_heatmap(correlation)
        html += "</div>\n"

    html += """
        </div>
    </body>
    </html>
    """

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding='utf-8')
    
    logger.info(f"Report exported to {out_path}")
    return out_path
