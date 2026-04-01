"""
Second Derivative Dashboard v2
Tracks acceleration across stagflation, AI infrastructure,
private credit, yield curve, prediction markets, and market structure.
"""

import dash
from dash import html, dcc, callback, Output, Input
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime

from data_engine import DashboardData

# ---------------------------------------------------------------------------
# Initialize
# ---------------------------------------------------------------------------

app = dash.Dash(
    __name__,
    title="2nd Derivative Dashboard",
    update_title="Updating...",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server

data = DashboardData()
try:
    data.refresh()
except Exception as e:
    print(f"Initial data refresh failed (will retry on first page load): {e}")

# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

COLORS = [
    '#58a6ff', '#3fb950', '#f85149', '#d29922',
    '#bc8cff', '#39d2c0', '#f778ba', '#79c0ff',
    '#ffa657', '#7ee787', '#ff7b72', '#d2a8ff',
]

BASE_LAYOUT = dict(
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif',
              size=11, color='#8b949e'),
    margin=dict(l=40, r=16, t=10, b=36),
    legend=dict(
        orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0,
        font=dict(size=10, color='#8b949e'), bgcolor='rgba(0,0,0,0)',
    ),
    xaxis=dict(gridcolor='#21262d', zeroline=False, showgrid=True, gridwidth=1),
    yaxis=dict(gridcolor='#21262d', zeroline=True, zerolinecolor='#484f58',
               zerolinewidth=1.5, showgrid=True, gridwidth=1),
    hovermode='x unified',
    hoverlabel=dict(bgcolor='#161b22', bordercolor='#30363d', font_size=11),
    height=300,
)


def make_chart(data_dict: dict, height: int = 300) -> go.Figure:
    """Build a multi-line chart from {name: pd.Series}."""
    fig = go.Figure()
    if not data_dict:
        fig.add_annotation(text="No data available", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(size=14, color='#6e7681'))
        fig.update_layout(**{**BASE_LAYOUT, 'height': height})
        return fig

    for i, (name, series) in enumerate(data_dict.items()):
        s = series.dropna().tail(90)
        if s.empty:
            continue
        color = COLORS[i % len(COLORS)]
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, name=name, mode='lines',
            line=dict(color=color, width=2),
            hovertemplate='%{y:.3f}<extra>' + name + '</extra>',
        ))

    fig.add_hline(y=0, line_dash="dot", line_color="#484f58", line_width=1)
    fig.update_layout(**{**BASE_LAYOUT, 'height': height})
    return fig


def make_zscore_chart(data_dict: dict) -> go.Figure:
    """Convergence chart with z-score bands."""
    fig = go.Figure()
    if not data_dict:
        fig.add_annotation(text="No data available", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(size=14, color='#6e7681'))
        fig.update_layout(**{**BASE_LAYOUT, 'height': 360})
        return fig

    for i, (name, series) in enumerate(data_dict.items()):
        s = series.dropna().tail(90)
        if s.empty:
            continue
        color = COLORS[i % len(COLORS)]
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, name=name, mode='lines',
            line=dict(color=color, width=2.5),
            hovertemplate='z=%{y:.2f}<extra>' + name + '</extra>',
        ))

    fig.add_hline(y=0, line_dash="dot", line_color="#484f58", line_width=1)
    fig.add_hline(y=1, line_dash="dash", line_color="rgba(248,81,73,0.3)", line_width=1,
                  annotation_text="+1\u03c3 Bearish", annotation_font_color="#f85149",
                  annotation_font_size=9, annotation_position="right")
    fig.add_hline(y=-1, line_dash="dash", line_color="rgba(63,185,80,0.3)", line_width=1,
                  annotation_text="-1\u03c3 Bullish", annotation_font_color="#3fb950",
                  annotation_font_size=9, annotation_position="right")
    fig.add_hline(y=2, line_dash="dot", line_color="rgba(248,81,73,0.15)", line_width=1)
    fig.add_hline(y=-2, line_dash="dot", line_color="rgba(63,185,80,0.15)", line_width=1)
    fig.add_hrect(y0=1, y1=3.5, fillcolor="rgba(248,81,73,0.04)", line_width=0)
    fig.add_hrect(y0=-3.5, y1=-1, fillcolor="rgba(63,185,80,0.04)", line_width=0)

    layout = {**BASE_LAYOUT, 'height': 360}
    layout['yaxis'] = {**layout['yaxis'], 'title': 'z-score', 'title_font_size': 10, 'range': [-3.5, 3.5]}
    fig.update_layout(**layout)
    return fig


def signal_chips(panel_data: dict) -> list:
    """Create signal indicator chips."""
    chips = []
    for name, series in panel_data.items():
        s = series.dropna()
        if s.empty:
            chips.append(html.Span(f"\u25cf {name}: N/A", className='signal-chip chip-neutral'))
            continue
        val = s.iloc[-1]
        if val > 0.5:
            cls, arrow = 'chip-bearish', '\u25b2'
        elif val < -0.5:
            cls, arrow = 'chip-bullish', '\u25bc'
        else:
            cls, arrow = 'chip-neutral', '\u2014'
        chips.append(html.Span(f"{arrow} {name}: {val:.2f}", className=f'signal-chip {cls}'))
    return chips


def prediction_cards(pm_data: dict) -> list:
    """Build prediction market probability cards grouped by category."""
    if not pm_data:
        return [html.Div("Prediction market data loading...",
                          style={'color': '#6e7681', 'fontSize': '13px'})]

    # Group by category
    groups = {}
    for name, info in pm_data.items():
        pct = info.get('yes_pct')
        if pct is None:
            continue
        cat = info.get('category', 'other')
        if cat not in groups:
            groups[cat] = []
        groups[cat].append((name, info))

    # Category display names and order
    cat_order = ['fed', 'economy', 'geopolitical', 'markets', 'politics',
                 'commodities', 'crypto', 'policy', 'discovered']
    cat_labels = {
        'fed': '\U0001f3e6 FED & RATES', 'economy': '\U0001f4c9 ECONOMY',
        'geopolitical': '\U0001f30d GEOPOLITICAL', 'markets': '\U0001f4c8 MARKETS',
        'politics': '\U0001f3db POLITICS', 'commodities': '\U0001f6e2 COMMODITIES',
        'crypto': '\u20bf CRYPTO', 'policy': '\U0001f4dc POLICY',
        'discovered': '\U0001f50d TRENDING',
    }

    sections = []
    for cat in cat_order:
        if cat not in groups:
            continue
        items = groups[cat]
        # Sort by how close to 50% (most uncertain = most interesting)
        items.sort(key=lambda x: abs(50 - x[1]['yes_pct']))

        cards = []
        for name, info in items:
            pct = info['yes_pct']
            q = info.get('question', name)

            if pct > 60:
                color, bg = '#f85149', 'rgba(248,81,73,0.08)'
            elif pct > 40:
                color, bg = '#d29922', 'rgba(210,153,34,0.08)'
            else:
                color, bg = '#3fb950', 'rgba(63,185,80,0.08)'

            cards.append(html.Div(
                className='prediction-card',
                style={'borderColor': color, 'background': bg},
                children=[
                    html.Div(f"{pct:.0f}%", className='prediction-pct', style={'color': color}),
                    html.Div(q[:55], className='prediction-question'),
                ]
            ))

        label = cat_labels.get(cat, cat.upper())
        sections.append(html.Div(className='prediction-section', children=[
            html.Div(label, className='prediction-section-label'),
            html.Div(className='prediction-scroll', children=cards),
        ]))

    return sections


def alert_items(alerts: list) -> list:
    """Build alert notification items."""
    items = []
    for a in alerts[:8]:  # Max 8 alerts
        sev = a.get('severity', 'medium')
        color = '#f85149' if sev == 'high' else '#d29922'
        icon = '\u26a0\ufe0f' if sev == 'high' else '\u26a1'
        items.append(html.Div(
            className='alert-item',
            style={'borderLeftColor': color},
            children=[
                html.Span(f"{icon} ", style={'marginRight': '4px'}),
                html.Span(a['name'], style={'fontWeight': '600', 'color': '#f0f6fc'}),
                html.Span(f" \u2014 {a['direction']} ({a['value']})",
                          style={'color': '#8b949e'}),
            ]
        ))
    if not items:
        items.append(html.Div("No alerts \u2014 all signals within normal range",
                              style={'color': '#3fb950', 'fontSize': '13px', 'padding': '8px 0'}))
    return items


# ---------------------------------------------------------------------------
# Helper: build a panel card
# ---------------------------------------------------------------------------

def panel(title, subtitle, icon, chart_id, full_width=False):
    cls = 'panel-card full-width' if full_width else 'panel-card'
    return html.Div(className=cls, children=[
        html.Div(className='panel-title', children=[
            html.Span(icon, className='panel-icon'), title,
        ]),
        html.Div(subtitle, className='panel-subtitle'),
        dcc.Graph(id=chart_id, config={'displayModeBar': False}),
    ])


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

app.layout = html.Div(id='app-container', children=[
    dcc.Interval(id='refresh-interval', interval=5 * 60 * 1000, n_intervals=0),

    # Header
    html.Div(className='dashboard-header', children=[
        html.H1("Second Derivative Dashboard"),
        html.Div("Tracking acceleration \u2014 how fast things are speeding up or slowing down",
                  className='subtitle'),
    ]),

    # Score + Alerts banner
    html.Div(className='top-banner', children=[
        html.Div(id='convergence-banner', className='convergence-banner'),
        html.Div(id='alerts-container', className='alerts-container', children=[
            html.Div("\u26a0\ufe0f Alerts", className='alerts-title'),
            html.Div(id='alert-items'),
        ]),
    ]),

    # Panel grid
    html.Div(className='panel-grid', children=[
        # Row 1: Convergence (full width)
        html.Div(className='panel-card full-width', children=[
            html.Div(className='panel-title', children=[
                html.Span("\u26a1", className='panel-icon'),
                "Convergence Monitor \u2014 The One Chart",
            ]),
            html.Div("5 key metrics normalized to z-scores. When all go red simultaneously, your thesis is expressing.",
                      className='panel-subtitle'),
            html.Div(id='convergence-signals', className='signal-row'),
            dcc.Graph(id='convergence-chart', config={'displayModeBar': False}),
        ]),

        # Row 2: Yield Curve + Private Credit
        panel("Yield Curve & Rates", "2/10 spread acceleration, real yields, credit spreads from FRED",
              "\U0001f4c8", 'yield-chart'),
        panel("Private Credit Stress", "Alt-asset managers, junk bonds, senior loans \u2014 acceleration of decline",
              "\U0001f3e6", 'credit-chart'),

        # Row 3: Oil/Inflation + AI Infra
        panel("Oil / Inflation Transmission", "Crude, gold, copper/gold ratio, agriculture, energy vs tech rotation",
              "\U0001f6e2\ufe0f", 'oil-chart'),
        panel("AI Infrastructure Cracks", "CoreWeave to NVDA supply chain \u2014 where the build is stressing",
              "\U0001f916", 'ai-chart'),

        # Row 4: FX + Market Structure
        panel("FX & Dollar Pressure", "USD/JPY, EUR, CNY, dollar index, trade-weighted dollar",
              "\U0001f4b1", 'fx-chart'),
        panel("Market Structure", "VIX term structure, defensive rotation, breadth, small vs large",
              "\U0001f4ca", 'structure-chart'),

        # Row 5: Prediction Markets (full width)
        html.Div(className='panel-card full-width', children=[
            html.Div(className='panel-title', children=[
                html.Span("\U0001f52e", className='panel-icon'),
                "Prediction Markets \u2014 Polymarket Live Probabilities",
            ]),
            html.Div("Track sudden shifts in implied probabilities \u2014 fast moves here often lead price action",
                      className='panel-subtitle'),
            html.Div(id='prediction-cards', className='prediction-grid'),
        ]),
    ]),

    # Footer
    html.Div(id='update-footer', className='update-footer'),
])


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@callback(
    Output('convergence-banner', 'children'),
    Output('convergence-signals', 'children'),
    Output('convergence-chart', 'figure'),
    Output('yield-chart', 'figure'),
    Output('credit-chart', 'figure'),
    Output('oil-chart', 'figure'),
    Output('ai-chart', 'figure'),
    Output('fx-chart', 'figure'),
    Output('structure-chart', 'figure'),
    Output('prediction-cards', 'children'),
    Output('alert-items', 'children'),
    Output('update-footer', 'children'),
    Input('refresh-interval', 'n_intervals'),
)
def update_dashboard(n):
    if n > 0 or data.last_updated is None:
        try:
            data.refresh()
        except Exception as e:
            print(f"Data refresh error: {e}")

    # --- Convergence banner ---
    bearish, total = data.convergence_score()
    if bearish >= 4:
        score_class, status = 'score-red', "FULL CONVERGENCE \u2014 All major signals aligned bearish"
    elif bearish >= 3:
        score_class, status = 'score-red', "HIGH ALERT \u2014 Multiple acceleration signals aligned"
    elif bearish >= 2:
        score_class, status = 'score-orange', "Elevated \u2014 Watch for additional confirmation"
    elif bearish >= 1:
        score_class, status = 'score-orange', "Single signal active \u2014 Not yet converging"
    else:
        score_class, status = 'score-green', "No convergence \u2014 Signals mixed or neutral"

    banner = [
        html.Div(className='score-box', children=[
            html.Div(f"{bearish}", className=f'score-number {score_class}'),
            html.Div("BEARISH", className='score-label'),
        ]),
        html.Div(className='score-divider'),
        html.Div(className='score-box', children=[
            html.Div(f"{total}", className='score-number'),
            html.Div("TOTAL", className='score-label'),
        ]),
        html.Div(className='score-divider'),
        html.Div(status, className='score-status'),
    ]

    # Build all panels
    conv_data = data.convergence_data()
    conv_signals = signal_chips(conv_data)
    conv_chart = make_zscore_chart(conv_data)

    yield_chart = make_chart(data.yield_curve_data())
    credit_chart = make_chart(data.private_credit_data())
    oil_chart = make_chart(data.oil_inflation_data())
    ai_chart = make_chart(data.ai_infra_data())
    fx_chart = make_chart(data.fx_dollar_data())
    struct_chart = make_chart(data.market_structure_data())

    pred_cards = prediction_cards(data.prediction_market_data())
    alerts = alert_items(data.get_all_alerts())

    ts = data.last_updated.strftime('%B %d, %Y at %I:%M %p') if data.last_updated else 'Never'
    footer = f"Last updated: {ts}  \u2022  Auto-refreshes every 5 minutes  \u2022  Data: Yahoo Finance + FRED + Polymarket"

    return (banner, conv_signals, conv_chart, yield_chart, credit_chart,
            oil_chart, ai_chart, fx_chart, struct_chart, pred_cards, alerts, footer)


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8050)
