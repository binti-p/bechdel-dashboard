# app.py  --  Bechdel dashboard (Plotly Dash)
# Needs bechdel_clean.csv in the same folder (made by make_bechdel_clean.py).
# Setup:  pip install dash pandas numpy plotly
# Run:    python app.py      then open  http://127.0.0.1:8050

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import sample_colorscale
from dash import Dash, dcc, html, Input, Output, State, ctx, no_update
from dash.exceptions import PreventUpdate


TITLE = "What types of films pass the Bechdel Test?"
HINT = ()
VIEW1_LABEL = "Bechdel results over time"
VIEW1_TITLE = "The passing share jumped from about a third to nearly half by the late 1990s, then stalled at 45–51%"
VIEW2_LABEL = "Cost and return"
VIEW2_TITLE = "Since the late-1990s, films that fail cost more to make, but earn back about the same per dollar"
VIEW3_LABEL = "Pass rate by genre"
VIEW3_TITLE = "Horror films pass more than twice as often as action or sci-fi films"

df = pd.read_csv("bechdel_clean.csv")
df["budget_m"] = df["budget_2013"] / 1e6          # budget in millions of 2013 dollars

PERIODS = ["1970–74", "1975–79", "1980–84", "1985–89", "1990–94",
           "1995–99", "2000–04", "2005–09", "2010–13"]
SUBTITLE = (f"{len(df):,} films, 1970–2013 · Bechdel results from FiveThirtyEight, "
            "genres from TMDB · dollars in 2013 terms")

# Bechdel levels, bottom of the stack first
LEVELS = ["ok", "men", "notalk", "nowomen", "dubious"]
NAMES = {"ok": "Passes", "men": "Men only", "notalk": "No talk", "nowomen": "<2 women", "dubious": "Dubious"}
COLORS = {"ok": "#08306B", "men": "#2171B5", "notalk": "#6BAED6", "nowomen": "#C6DBEF", "dubious": "#707070"}

MIN_LINE = 5      # fewer films than this: draw nothing
LOW_N = 15        # fewer films than this: hollow marker / star
DIM = 0.3         # opacity of things that are not selected

SCALE = [[0, "#FFFFFF"], [0.2, "#DEEBF7"], [0.4, "#9ECAE1"], [0.6, "#4292C6"], [0.8, "#08519C"], [1, "#000814"]]


def filter_data(data, genre=None, periods=None):
    """Keep only the films that match the current selection."""
    if genre:
        data = data[data["genre_group"] == genre]
    if periods:
        data = data[data["period"].isin(periods)]
    return data


def periods_axis(selected_periods):
    """Which periods to put on the X axis: the selected ones (in canonical order), or all."""
    if selected_periods:
        return [p for p in PERIODS if p in selected_periods]
    return list(PERIODS)


def periods_from_selection(selection, axis):
    """Turn a drag-selection on the period axis into a list of periods ([] = no filter).
    Returns None when the selection is empty because the chart was just redrawn (ignore it)."""
    if not selection:                      # double-click clears the selection
        return []
    chosen = []
    box = selection.get("range")
    x_range = next((v for k, v in box.items() if k.startswith("x")), None) if box else None
    if x_range:
        # on this axis period number i sits at position i (0, 1, 2, ...)
        left, right = min(x_range), max(x_range)
        chosen = [p for i, p in enumerate(axis) if i + 0.4 > left and i - 0.4 < right]
    else:                                  # a single click on a bar / marker
        points = selection.get("points", [])
        if not points:
            return None
        chosen = [point["x"] for point in points]
    chosen = [p for p in PERIODS if p in chosen]
    return [] if len(chosen) == len(PERIODS) else chosen


def empty_figure(message):
    fig = go.Figure()
    fig.add_annotation(text=message, xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                       font=dict(size=14, color="#666"))
    fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False), plot_bgcolor="white", height=300)
    return fig


# VIEW 1: Bechdel mix over time (100% stacked bars)
def make_view1(data, selected_periods):
    axis = periods_axis(selected_periods)
    counts = pd.crosstab(data["period"], data["clean_test"])
    counts = counts.reindex(index=axis, columns=LEVELS, fill_value=0)
    n = counts.sum(axis=1)
    pct = counts.div(n.replace(0, np.nan), axis=0) * 100
    pct.loc[n < MIN_LINE] = np.nan                       # too few films: no bar

    fig = go.Figure()
    for level in LEVELS:
        fig.add_bar(x=axis, y=pct[level], name=NAMES[level], marker_color=COLORS[level],
                    marker_line=dict(color="white", width=1),
                    customdata=counts[level],
                    hovertemplate="%{x}<br>" + NAMES[level] + ": %{y:.0f}% (%{customdata} films)<extra></extra>")

    # n above every bar (star = fewer than 15 films)
    for p in axis:
        star = "*" if n[p] < LOW_N else ""
        fig.add_annotation(x=p, y=103, text=f"n={n[p]}{star}", showarrow=False, yanchor="bottom",
                           font=dict(size=11, color="#555"))

    fig.update_layout(barmode="stack", showlegend=True, height=380, plot_bgcolor="white",
                      margin=dict(l=55, r=130, t=30, b=50),
                      legend=dict(orientation="v", yanchor="top", y=1.0, xanchor="left", x=1.01,
                                  font=dict(size=11), bgcolor="rgba(0,0,0,0)",
                                  traceorder="normal", itemclick=False, itemdoubleclick=False),
                      dragmode="select", selectdirection="h")      # drag = pick years only (no zoom)
    fig.update_yaxes(title="Share of films", ticksuffix="%", range=[0, 112], gridcolor="#eee", fixedrange=True)
    fig.update_xaxes(title="Release period (5 years)", categoryorder="array", categoryarray=axis, fixedrange=True)
    return fig


# VIEW 2: cost and return (two line panels sharing the period axis)
def line_data(data, group, column, axis):
    """Median of `column` and number of films, for each period in `axis`."""
    d = data[data["pass_group"] == group].dropna(subset=[column])
    t = d.groupby("period")[column].agg(n="size", median="median").reindex(axis)
    t["n"] = t["n"].fillna(0)
    t.loc[t["n"] < MIN_LINE, "median"] = np.nan          # too few films: no point, no line
    return t


def make_view2(data, context, selected_periods, show_context, full_data, full_context):
    axis = periods_axis(selected_periods)
    style = {"Pass": ("#08306B", "circle"), "Not pass": ("#6BAED6", "triangle-up")}

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.16,
                        subplot_titles=("What studios spend: median budget",
                                        "What they get back: median return per dollar"))
    # work out the median tables first (we need them to place the end labels)
    panels = [(1, "budget_m", "$M"), (2, "roi", "×")]
    tables = {(row, group): line_data(data, group, column, axis)
              for row, column, unit in panels for group in ["Not pass", "Pass"]}

    # y-axis ranges use every period (and the genre filter), so choosing years does not change them
    budget_medians, all_medians = [], []
    for row, column, unit in panels:
        for group in ["Not pass", "Pass"]:
            for source in ([full_data, full_context] if show_context else [full_data]):
                medians = list(line_data(source, group, column, PERIODS)["median"].dropna())
                (budget_medians if row == 1 else all_medians).extend(medians)

    legend_names = {"Pass": "Passes Bechdel", "Not pass": "Does not pass"}
    for row, column, unit in panels:
        for group in ["Not pass", "Pass"]:
            color, symbol = style[group]

            # faded lines for all genres, so a genre can be compared with the rest
            if show_context:
                c = line_data(context, group, column, axis)
                fig.add_trace(go.Scatter(x=axis, y=c["median"], mode="lines", opacity=0.25,
                                         line=dict(color=color, width=3), hoverinfo="skip",
                                         showlegend=False), row=row, col=1)

            s = tables[(row, group)]
            hollow = (s["n"] < LOW_N).tolist()

            fig.add_trace(go.Scatter(
                x=axis, y=s["median"], mode="lines+markers",
                name=legend_names[group], legendgroup=group,
                showlegend=(row == 1),      # one legend entry per group, from the top panel
                cliponaxis=False,
                line=dict(color=color, width=3.5),
                marker=dict(symbol=symbol, size=11,
                            color=["white" if h else color for h in hollow], line=dict(color=color, width=2)),
                customdata=s["n"],
                hovertemplate=group + "<br>%{x}<br>Median: %{y:.1f}" + unit +
                              "<br>n = %{customdata:.0f} films<extra></extra>"), row=row, col=1)

    # break-even line (1x) in the bottom panel
    fig.add_trace(go.Scatter(x=axis, y=[1] * len(axis), mode="lines+text", showlegend=False,
                             line=dict(color="#888", width=1, dash="dash"), hoverinfo="skip",
                             text=["Break-even (1×)"] + [""] * (len(axis) - 1),
                             textposition="top right", textfont=dict(size=11, color="#666")), row=2, col=1)

    # bottom axis: log scale, wide enough for whatever is shown
    top = max([10] + [m * 1.4 for m in all_medians])
    bottom = min([0.5] + [m / 1.4 for m in all_medians])
    ticks = [t for t in [0.1, 0.3, 1, 3, 10, 30, 100] if bottom <= t <= top]
    budget_top = max(budget_medians) * 1.15 if budget_medians else 1
    fig.update_yaxes(title="Budget (2013 $M)", range=[0, budget_top], gridcolor="#eee", row=1, col=1)
    fig.update_yaxes(title="Gross ÷ budget (log)", type="log", range=[np.log10(bottom), np.log10(top)],
                     tickvals=ticks, ticktext=[f"{t:g}×" for t in ticks], gridcolor="#eee", row=2, col=1)
    fig.update_yaxes(fixedrange=True)                      # y never zooms
    fig.update_xaxes(categoryorder="array", categoryarray=axis, fixedrange=True)
    fig.update_xaxes(title="Release period (5 years)", row=2, col=1)
    fig.update_layout(height=560, plot_bgcolor="white", margin=dict(l=65, r=140, t=40, b=50),
                      legend=dict(orientation="v", yanchor="top", y=1.0, xanchor="left", x=1.01,
                                  font=dict(size=11), bgcolor="rgba(0,0,0,0)",
                                  itemclick=False, itemdoubleclick=False),
                      dragmode="select", selectdirection="h")      # drag = pick years only (no zoom)
    return fig


# VIEW 3: pass rate by genre (band = 95% interval, dot = pass rate)
def make_view3(data, selected_genre):
    g = data.groupby("genre_group").agg(n=("clean_test", "size"),
                                         k=("clean_test", lambda s: (s == "ok").sum())).reset_index()
    if len(g) == 0:
        return empty_figure("No films match this selection")
    g["p"] = g["k"] / g["n"]

    # 95% Wilson interval
    z = 1.96
    den = 1 + z**2 / g["n"]
    centre = (g["p"] + z**2 / (2 * g["n"])) / den
    half = z * np.sqrt(g["p"] * (1 - g["p"]) / g["n"] + z**2 / (4 * g["n"]**2)) / den
    g["lo"], g["hi"] = centre - half, centre + half

    # highest pass rate first, "Other" always last
    g = pd.concat([g[g["genre_group"] != "Other"].sort_values("p", ascending=False),
                   g[g["genre_group"] == "Other"]]).reset_index(drop=True)
    g["label"] = g["genre_group"] + "  (n=" + g["n"].astype(str) + ")"
    overall = g["k"].sum() / g["n"].sum()

    # selected genre stays solid, the rest fade
    opacity = [1.0 if selected_genre in (None, name) else DIM for name in g["genre_group"]]
    shade = sample_colorscale(SCALE, list(g["p"]))
    hollow = (g["n"] < LOW_N).tolist()

    fig = go.Figure()
    fig.add_bar(y=g["label"], x=(g["hi"] - g["lo"]) * 100, base=g["lo"] * 100, orientation="h", width=0.66,
                marker=dict(color=shade, opacity=opacity, line=dict(color="#333", width=1)),
                hoverinfo="skip", showlegend=False)
    fig.add_scatter(y=g["label"], x=g["p"] * 100, mode="markers", showlegend=False,
                    marker=dict(size=10, opacity=opacity,
                                color=["white" if h else "#111" for h in hollow],
                                line=dict(color=["#111" if h else "white" for h in hollow], width=2)),
                    customdata=np.c_[g["n"], g["k"], g["lo"] * 100, g["hi"] * 100],
                    hovertemplate="%{y}<br>Passes: %{x:.0f}% (%{customdata[1]:.0f} of %{customdata[0]:.0f} films)"
                                  "<br>95% interval: %{customdata[2]:.0f}% to %{customdata[3]:.0f}%<extra></extra>")
    # the % printed to the right of each band
    for label, hi, p, o in zip(g["label"], g["hi"], g["p"], opacity):
        fig.add_annotation(x=hi * 100, y=label, text=f"<b>{p:.0%}</b>", xanchor="left", xshift=8,
                           showarrow=False, font=dict(size=13, color="#222"), opacity=o,
                           bgcolor="white", borderpad=2)

    # dashed line = pass rate of all the films shown
    fig.add_vline(x=overall * 100, line=dict(color="#222", width=1.5, dash="dash"))
    fig.add_annotation(x=overall * 100, y=1.0, yref="paper", yanchor="bottom", xanchor="center",
                       showarrow=False, text=f"All films: {overall:.0%}", font=dict(size=12, color="#222"))

    fig.update_layout(height=110 + 36 * len(g), plot_bgcolor="white", margin=dict(l=140, r=20, t=40, b=60),
                      xaxis=dict(title="Share of films that pass", range=[0, 100], ticksuffix="%", gridcolor="#eee"),
                      yaxis=dict(autorange="reversed", categoryorder="array", categoryarray=list(g["label"])))
    return fig


# PAGE LAYOUT
app = Dash(__name__)
server = app.server

CARD = {"background": "white", "border": "1px solid #d9d9d9", "borderRadius": "8px",
        "padding": "12px 14px", "marginBottom": "14px"}
LABEL = {"fontSize": "12px", "color": "#666", "textTransform": "uppercase", "letterSpacing": "0.05em"}
HEADING = {"fontSize": "17px", "fontWeight": "600", "margin": "2px 0 8px 0", "color": "#08306B"}
NOTE = {"fontSize": "12px", "color": "#666", "margin": "4px 0 0 0"}

# What the five Bechdel categories mean. Used in the glossary card above View 3.
GLOSSARY = [
    ("Passes",    "meets all three rules above"),
    ("Men only",  "has two or more named women, but they only talk about men"),
    ("No talk",   "has two or more named women, but they never talk to each other"),
    ("<2 women",  "fewer than two named women in the film"),
    ("Dubious",   "sources disagree on whether it passes (counted as not passing here)"),
]


def card(label, title, graph_id, note):
    return html.Div(style=CARD, children=[
        html.Div(label, style=LABEL),
        html.H3(title, style=HEADING),
        dcc.Graph(id=graph_id, config={"displayModeBar": False}),
        html.P(note, style=NOTE),
    ])


app.layout = html.Div(
    style={"fontFamily": "Arial, sans-serif", "background": "#f3f5f8", "padding": "16px 22px"},
    children=[
        dcc.Store(id="state", data={"periods": [], "genre": None}),

        html.H1(TITLE, style={"fontSize": "24px", "margin": "0", "color": "#08306B"}),
        html.Div(SUBTITLE, style={"color": "#555", "margin": "4px 0 12px 0"}),

        # filter strip: hint on top, reset button next to the "Showing..." line
        html.Div(style={**CARD},
                 children=[
                     html.Div(HINT, style={"color": "#666", "fontSize": "13px", "marginBottom": "8px"}),
                     html.Div(style={"display": "flex", "alignItems": "center", "gap": "12px"},
                              children=[
                                  html.Button("Reset", id="reset", n_clicks=0),
                                  html.Div(id="status", style={"fontWeight": "600", "color": "#08306B"}),
                              ]),
                 ]),

        # left column: View 1 on top of View 2 (same period axis); right column: View 3
        html.Div(style={"display": "grid", "gridTemplateColumns": "1.5fr 1fr", "gap": "14px",
                        "alignItems": "start"},
                 children=[
                     html.Div([
                         card(VIEW1_LABEL, VIEW1_TITLE, "view1",
                              "Dubious counts as not passing. * = fewer than 15 films; bars are hidden under 5 films. "
                              "Drag across periods to filter every view to those years; double-click to clear."),
                         card(VIEW2_LABEL, VIEW2_TITLE, "view2",
                              "Medians per 5-year period. Hollow marker = fewer than 15 films; no line under 5. "
                              "Dubious counts as not passing. Gross is worldwide, not profit. All dollars are 2013 dollars. "
                              "Faded lines = all genres (shown when a genre is selected)."),
                     ]),
                     html.Div([
                         # own card, sits above view 3
                         html.Div(style=CARD, children=[
                             html.Div("About the Bechdel test", style=LABEL),
                             html.Div(style={"marginTop": "6px", "fontSize": "13px",
                                             "color": "#2b2b2b", "lineHeight": "1.5"}, children=[
                                 html.Div([
                                     "A film passes if it has (1) at least two named women who (2) talk to "
                                     "each other (3) about something besides a man. It's a rough proxy for "
                                     "how prominent women are on screen, not a judgement of the film.",
                                 ]),
                                 html.Div(style={"marginTop": "8px"}, children=[html.B("Category meanings:")]),
                                 html.Ul(style={"margin": "4px 0 0 18px", "padding": 0}, children=[
                                     html.Li([html.B(name + ": "), desc]) for name, desc in GLOSSARY
                                 ]),
                             ]),
                         ]),
                         card(VIEW3_LABEL, VIEW3_TITLE, "view3",
                              "Dot and % = share that passes; band = 95% interval; darker shade = higher pass "
                              "rate (white = 0%, black = 100%). Hollow dot = fewer than 15 films. Genres under "
                              "20 films are grouped as Other. Click a genre to filter Views 1 and 2."),
                     ]),
                 ]),
    ])


# INTERACTION
@app.callback(
    Output("state", "data"), Output("view3", "clickData"),
    Input("view1", "selectedData"), Input("view2", "selectedData"), Input("view3", "clickData"),
    Input("reset", "n_clicks"),
    State("state", "data"),
    prevent_initial_call=True,
)
def update_state(select1, select2, click3, reset_clicks, state):
    """Remember which periods / genre are selected."""
    periods = list(state["periods"])
    genre = state["genre"]
    trigger = ctx.triggered_id

    if trigger == "reset":
        periods, genre = [], None
    elif trigger in ("view1", "view2"):
        selection = select1 if trigger == "view1" else select2
        # position 0, 1, 2... on the axis = the periods currently shown
        new_periods = periods_from_selection(selection, periods_axis(periods))
        if new_periods is None:
            raise PreventUpdate
        periods = new_periods
    elif trigger == "view3":
        if click3 is None:
            raise PreventUpdate
        name = click3["points"][0]["y"].split("  (n=")[0]      # "Horror  (n=134)" -> "Horror"
        genre = None if genre == name else name
    else:
        raise PreventUpdate

    # view 3's clickData is set back to None so the same bar can be clicked again
    return {"periods": periods, "genre": genre}, None


@app.callback(
    Output("view1", "figure"), Output("view2", "figure"), Output("view3", "figure"), Output("status", "children"),
    Input("state", "data"),
)
def draw(state):
    """Redraw all three views from the current selection."""
    periods, genre = state["periods"], state["genre"]

    # Years zoom every view. Genre filters views 1 & 2 (view 3 shows every genre).
    for_views_1_2 = filter_data(df, genre=genre, periods=periods)
    for_view_3 = filter_data(df, periods=periods)
    context_1_2 = filter_data(df, periods=periods)            # "all genres" reference lines for view 2

    fig1 = make_view1(for_views_1_2, periods)
    fig2 = make_view2(for_views_1_2, context_1_2, periods, genre is not None,
                      full_data=filter_data(df, genre=genre), full_context=df)
    fig3 = make_view3(for_view_3, genre)

    parts = [genre + " films" if genre else "all genres"]
    parts.append("years: " + ", ".join(sorted(periods)) if periods else "all years")
    status = "Showing: " + " · ".join(parts)
    return fig1, fig2, fig3, status


if __name__ == "__main__":
    app.run(debug=False)