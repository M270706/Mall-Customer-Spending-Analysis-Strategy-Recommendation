import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.data_validation import (
    SEGMENT_STRATEGIES,
    SPENDING_SCORE_BINS,
    SPENDING_SEGMENTS,
    validate_customer_data,
)

SEGMENT_ORDER = list(SPENDING_SEGMENTS)
SEGMENT_COLORS = {"Low": "#8fa3bf", "Medium": "#e6a84b", "High": "#40b89b"}

st.set_page_config(
    page_title="Grahak Gyaan",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --ink: #17283f;
        --muted: #718198;
        --line: #e7edf3;
        --surface: #ffffff;
        --canvas: #f5f7fa;
        --teal: #178f83;
    }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: var(--canvas); color: var(--ink); }
    [data-testid="stAppViewContainer"], [data-testid="stMain"] { background: var(--canvas); }
    [data-testid="stHeader"] { background: rgba(245,247,250,.92); }
    [data-testid="stSidebar"] {
        background: #fff;
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.2rem; }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p { color: var(--ink) !important; }
    [data-baseweb="select"] > div,
    [data-baseweb="input"] > div {
        background: #fff !important;
        color: var(--ink) !important;
        border-color: #d7e0e9 !important;
    }
    [data-baseweb="select"] input,
    [data-baseweb="input"] input { color: var(--ink) !important; }
    [data-baseweb="tag"] {
        background: #e6f4f1 !important;
        color: #167b71 !important;
    }
    [data-baseweb="tag"] span { color: #167b71 !important; }
    [data-testid="stWidgetLabel"] p { color: var(--muted) !important; }
    [data-testid="stSlider"] [role="slider"] { background: var(--teal) !important; }
    .block-container { max-width: 1440px; padding-top: 4rem; padding-bottom: 3rem; }
    h1, h2, h3 { font-family: 'Manrope', sans-serif; color: var(--ink); letter-spacing: -.035em; }
    h1 { font-size: 2.15rem !important; font-weight: 800 !important; margin-bottom: .15rem !important; }
    h1 [data-testid="stHeaderActionElements"] { display: none !important; }
    h2 { font-size: 1.25rem !important; font-weight: 700 !important; }
    h3 { font-size: 1.03rem !important; font-weight: 700 !important; }
    [data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0 3px 12px rgba(30,55,80,.035);
    }
    [data-testid="stMetricLabel"] p { color: var(--muted); font-size: .82rem; font-weight: 600; }
    [data-testid="stMetricValue"] { color: var(--ink); font-family: 'Manrope', sans-serif; font-weight: 800; font-size: 1.65rem; }
    [data-testid="stPlotlyChart"], [data-testid="stDataFrame"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 4px;
    }
    [data-testid="stPlotlyChart"] {
        padding: 0;
        overflow: hidden;
    }
    .eyebrow {
        display: inline-flex; align-items: center;
        color: #0f625c !important; background: #e7f2f0; border: 1px solid #c7dfdb;
        border-radius: 999px; padding: .28rem .65rem;
        font-family: 'Manrope', sans-serif; font-weight: 800;
        font-size: .76rem; line-height: 1.4;
        text-transform: uppercase; letter-spacing: .1em; margin-bottom: .65rem;
    }
    .subtitle { color: var(--muted); font-size: .98rem; margin-bottom: 1.45rem; }
    [data-testid="stCaptionContainer"] p { color: var(--muted) !important; }
    .section-label {
        color: var(--muted); text-transform: uppercase; letter-spacing: .1em;
        font-size: .72rem; font-weight: 700; margin: .25rem 0 .55rem;
    }
    .chart-legend {
        display: flex; flex-wrap: wrap; gap: .45rem 1.15rem;
        align-items: center; margin: .1rem 0 .45rem; color: var(--muted);
        font-size: .86rem; font-weight: 600;
    }
    .chart-legend-item { display: inline-flex; align-items: center; gap: .42rem; }
    .chart-legend-dot { width: .72rem; height: .72rem; border-radius: 50%; display: inline-block; }
    .insight-card {
        background: #fff; border: 1px solid var(--line); border-left: 4px solid var(--teal);
        border-radius: 12px; padding: 16px 18px; min-height: 100px;
    }
    .insight-title { color: var(--muted); font-size: .78rem; font-weight: 600; margin-bottom: 5px; }
    .insight-value { color: var(--ink); font-family: 'Manrope',sans-serif; font-size: 1.15rem; font-weight: 800; }
    .insight-note { color: var(--muted); font-size: .78rem; margin-top: 3px; }
    .brand-mark {
        font-family: 'Manrope',sans-serif; color: var(--ink); font-weight: 800;
        font-size: 1.03rem; padding: .2rem 0 1.1rem;
    }
    .brand-dot { color: var(--teal); }
    div[data-testid="stRadio"] > div { gap: .45rem; }
    div[data-testid="stRadio"] label {
        border: 1px solid var(--line); border-radius: 9px; padding: .35rem .7rem;
        background: #fff;
    }
    div[data-testid="stRadio"] label,
    div[data-testid="stRadio"] label p { color: var(--ink) !important; }
    div[data-testid="stRadio"] [role="radio"][aria-checked="true"] {
        background-color: var(--teal) !important;
        border-color: var(--teal) !important;
    }
    @media (max-width: 800px) {
        .block-container { padding: 4rem 1rem 2rem; }
        h1 { font-size: 1.7rem !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data() -> pd.DataFrame:
    data_path = PROJECT_ROOT / "data" / "Mall_Customers.csv"
    if not data_path.is_file():
        raise FileNotFoundError(f"Customer dataset not found: {data_path}")

    df = validate_customer_data(pd.read_csv(data_path))

    df["Spending_Category"] = pd.cut(
        df["Spending Score (1-100)"],
        bins=SPENDING_SCORE_BINS,
        labels=SEGMENT_ORDER,
        right=False,
        include_lowest=True,
    )
    df["Recommended_Strategy"] = df["Spending_Category"].map(SEGMENT_STRATEGIES)
    return df


def style_chart(fig, height: int = 360):
    fig.update_layout(
        height=height,
        margin=dict(l=12, r=12, t=52, b=48),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color="#64748b", size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.16,
            xanchor="left",
            x=0,
            title_text="",
            font=dict(color="#64748b"),
        ),
        title=dict(
            text=fig.layout.title.text,
            font=dict(family="Manrope, sans-serif", color="#17283f", size=15),
            x=0.02,
            xanchor="left",
            y=0.98,
            yanchor="top",
            automargin=True,
        ),
    )
    fig.update_xaxes(
        showgrid=False,
        linecolor="#e7edf3",
        zeroline=False,
        tickfont=dict(color="#64748b"),
        title_font=dict(color="#64748b"),
        tickangle=0,
    )
    fig.update_yaxes(
        gridcolor="#edf1f5",
        zeroline=False,
        tickfont=dict(color="#64748b"),
        title_font=dict(color="#64748b"),
    )
    return fig


def insight_card(title: str, value: str, note: str) -> str:
    return (
        '<div class="insight-card">'
        f'<div class="insight-title">{title}</div>'
        f'<div class="insight-value">{value}</div>'
        f'<div class="insight-note">{note}</div>'
        "</div>"
    )


try:
    customers = load_data()
except (FileNotFoundError, ValueError) as error:
    st.error(f"Unable to load the customer dataset: {error}")
    st.stop()

with st.sidebar:
    st.markdown('<div class="brand-mark">GRAHAK<span class="brand-dot"> GYAAN</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Narrow your view</div>', unsafe_allow_html=True)
    st.caption("Choose a few filters to update the charts and customer list.")

    age_range = st.slider(
        "Age range",
        min_value=int(customers["Age"].min()),
        max_value=int(customers["Age"].max()),
        value=(int(customers["Age"].min()), int(customers["Age"].max())),
        help="Show customers within this age range.",
    )
    income_range = st.slider(
        "Annual income range (k$)",
        min_value=int(customers["Annual Income (k$)"].min()),
        max_value=int(customers["Annual Income (k$)"].max()),
        value=(
            int(customers["Annual Income (k$)"].min()),
            int(customers["Annual Income (k$)"].max()),
        ),
        help="Annual income is shown in thousands of dollars.",
    )
    genders = st.multiselect(
        "Gender",
        options=sorted(customers["Gender"].dropna().unique().tolist()),
        default=sorted(customers["Gender"].dropna().unique().tolist()),
    )
    categories = st.multiselect(
        "Spending score group",
        options=SEGMENT_ORDER,
        default=SEGMENT_ORDER,
        help="Groups use the dataset's spending score: Low below 40, Medium 40–69, and High 70 or above.",
    )
    st.divider()
    st.caption(f"{len(customers):,} customer records")
    st.caption("Dataset: Mall_Customers.csv")

filtered = customers[
    customers["Age"].between(*age_range)
    & customers["Annual Income (k$)"].between(*income_range)
    & customers["Gender"].isin(genders)
    & customers["Spending_Category"].isin(categories)
].copy()

st.markdown('<div class="eyebrow">Grahak Gyaan · Customer analysis</div>', unsafe_allow_html=True)
st.title("A closer look at your customers.")
st.markdown(
    '<div class="subtitle">Explore customer profiles, compare spending-score groups, and see how the dataset is distributed.</div>',
    unsafe_allow_html=True,
)

page = st.radio(
    "Choose a view",
    ["At a glance", "Compare groups", "Customer list"],
    horizontal=True,
    label_visibility="collapsed",
)

if filtered.empty:
    st.info("No customers match these filters. Widen your age, income, gender, or segment selection.")
    st.stop()

customer_count = len(filtered)
avg_age = filtered["Age"].mean()
avg_income = filtered["Annual Income (k$)"].mean()
avg_score = filtered["Spending Score (1-100)"].mean()
high_share = filtered["Spending_Category"].eq("High").mean() * 100

if page == "At a glance":
    metric_cols = st.columns(4)
    metric_cols[0].metric("Customers", f"{customer_count:,}", help=f"Filtered from {len(customers):,} total customers.")
    metric_cols[1].metric("Avg. age", f"{avg_age:.1f}", help="Average customer age in years.")
    metric_cols[2].metric("Avg. income", f"${avg_income:.1f}k", help="Average annual income, in thousands of dollars.")
    metric_cols[3].metric("Avg. score", f"{avg_score:.1f}", help="Average spending score, out of 100.")

    chart_left, chart_right = st.columns([1, 1.65], gap="large")
    with chart_left:
        counts = (
            filtered["Spending_Category"]
            .value_counts()
            .reindex(SEGMENT_ORDER, fill_value=0)
            .rename_axis("Segment")
            .reset_index(name="Customers")
        )
        fig_mix = px.pie(
            counts,
            names="Segment",
            values="Customers",
            hole=0.68,
            color="Segment",
            color_discrete_map=SEGMENT_COLORS,
            title="Segment mix",
        )
        fig_mix.update_traces(
            textposition="inside",
            textinfo="percent",
            hovertemplate="%{label}<br>%{value} customers (%{percent})<extra></extra>",
        )
        fig_mix.add_annotation(
            text=f"<b>{customer_count:,}</b><br><span style='font-size:11px'>customers</span>",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=19, color="#17283f"),
        )
        fig_mix.update_layout(showlegend=False)
        st.plotly_chart(style_chart(fig_mix), width="stretch", config={"displayModeBar": False})

    with chart_right:
        color_by = st.selectbox("Group points by", ["Spending score group", "Gender"], key="overview_color")
        color_column = "Spending_Category" if color_by == "Spending score group" else "Gender"
        color_map = SEGMENT_COLORS if color_by == "Spending score group" else {"Female": "#8e79c6", "Male": "#4c8ebf"}
        legend_labels = SEGMENT_ORDER if color_by == "Spending score group" else ["Female", "Male"]
        legend_items = "".join(
            '<span class="chart-legend-item">'
            f'<span class="chart-legend-dot" style="background:{color_map[label]}"></span>'
            f"{label}</span>"
            for label in legend_labels
        )
        st.markdown(
            f'<div class="chart-legend" aria-label="{color_by} legend">{legend_items}</div>',
            unsafe_allow_html=True,
        )
        fig_scatter = px.scatter(
            filtered,
            x="Annual Income (k$)",
            y="Spending Score (1-100)",
            color=color_column,
            color_discrete_map=color_map,
            size="Age",
            size_max=17,
            hover_name="CustomerID",
            hover_data={
                "Gender": True,
                "Age": True,
                "Annual Income (k$)": ":.0f",
                "Spending Score (1-100)": True,
                "Spending_Category": True,
                "CustomerID": False,
            },
            title="Annual income and spending score",
            labels={
                "Annual Income (k$)": "Annual income (k$)",
                "Spending Score (1-100)": "Spending score",
                "Spending_Category": "Segment",
            },
            category_orders={color_column: legend_labels},
            opacity=0.82,
        )
        styled_scatter = style_chart(fig_scatter)
        styled_scatter.update_layout(showlegend=False)
        st.plotly_chart(styled_scatter, width="stretch", config={"displayModeBar": False})

    st.subheader("A few quick observations")
    leaders = filtered["Spending_Category"].value_counts()
    largest_segment = leaders.idxmax() if not leaders.empty else "N/A"
    largest_count = int(leaders.max()) if not leaders.empty else 0
    insight_cols = st.columns(3)
    insight_cols[0].markdown(
        insight_card("Largest score group", largest_segment, f"{largest_count:,} customers in this view"),
        unsafe_allow_html=True,
    )
    insight_cols[1].markdown(
        insight_card("Customers in the high group", f"{high_share:.1f}%", "Spending score of 70 or above"),
        unsafe_allow_html=True,
    )
    insight_cols[2].markdown(
        insight_card("Average customer age", f"{avg_age:.1f} years", f"Across {customer_count:,} filtered customers"),
        unsafe_allow_html=True,
    )

elif page == "Compare groups":
    st.subheader("Compare spending-score groups")
    st.caption("These groups describe the score in this dataset; they are not measures of customer lifetime value.")

    segment_counts = (
        filtered["Spending_Category"]
        .value_counts()
        .reindex(SEGMENT_ORDER, fill_value=0)
        .rename_axis("Segment")
        .reset_index(name="Customers")
    )
    selected_segment = st.selectbox(
        "Choose a group",
        options=SEGMENT_ORDER,
        index=SEGMENT_ORDER.index(segment_counts.loc[segment_counts["Customers"].idxmax(), "Segment"])
        if segment_counts["Customers"].max() > 0
        else 0,
    )

    left, right = st.columns([1, 1.4], gap="large")
    with left:
        fig_segments = px.bar(
            segment_counts,
            x="Segment",
            y="Customers",
            color="Segment",
            color_discrete_map=SEGMENT_COLORS,
            title="Customers per score group",
            text="Customers",
            category_orders={"Segment": SEGMENT_ORDER},
        )
        fig_segments.update_traces(textposition="outside", marker_line_width=0)
        fig_segments.update_layout(showlegend=False)
        st.plotly_chart(style_chart(fig_segments, 360), width="stretch", config={"displayModeBar": False})
    with right:
        fig_profile = px.box(
            filtered,
            x="Spending_Category",
            y="Age",
            color="Spending_Category",
            color_discrete_map=SEGMENT_COLORS,
            points="outliers",
            title="Age range within each group",
            labels={"Spending_Category": "Segment", "Age": "Customer age (years)"},
            category_orders={"Spending_Category": SEGMENT_ORDER},
        )
        fig_profile.update_layout(showlegend=False)
        st.plotly_chart(style_chart(fig_profile, 360), width="stretch", config={"displayModeBar": False})

    chosen = filtered[filtered["Spending_Category"] == selected_segment]
    if chosen.empty:
        st.info(f"No {selected_segment.lower()}-segment customers match the current filters.")
    else:
        st.markdown(f'<div class="section-label">{selected_segment} group at a glance</div>', unsafe_allow_html=True)
        stats = st.columns(4)
        stats[0].metric("Customers", f"{len(chosen):,}")
        stats[1].metric("Average age", f"{chosen['Age'].mean():.1f}")
        stats[2].metric("Average income (k$)", f"${chosen['Annual Income (k$)'].mean():.1f}")
        stats[3].metric("Average score", f"{chosen['Spending Score (1-100)'].mean():.1f}/100")
        st.caption(
            "An idea to consider: "
            f"{chosen['Recommended_Strategy'].iloc[0]}. It is based only on the spending-score group and has not been tested."
        )

else:
    st.subheader("Customer list")
    st.caption("Search, sort, or download the records that match your filters.")
    search = st.text_input("Find a customer ID", placeholder="For example, 108")
    directory = filtered.copy()
    if search.strip():
        directory = directory[
            directory["CustomerID"].astype(str).str.contains(search.strip(), regex=False, case=False)
        ]

    sort_column = st.selectbox(
        "Sort by",
        ["CustomerID", "Age", "Annual Income (k$)", "Spending Score (1-100)", "Spending_Category"],
    )
    sort_direction = st.toggle("Reverse order", value=False)
    directory = directory.sort_values(sort_column, ascending=not sort_direction)
    display_columns = [
        "CustomerID",
        "Gender",
        "Age",
        "Annual Income (k$)",
        "Spending Score (1-100)",
        "Spending_Category",
        "Recommended_Strategy",
    ]
    st.dataframe(
        directory[display_columns],
        width="stretch",
        hide_index=True,
        height=470,
        column_config={
            "CustomerID": st.column_config.TextColumn("Customer ID"),
            "Annual Income (k$)": st.column_config.NumberColumn("Annual income (k$)", format="$%d"),
            "Spending Score (1-100)": st.column_config.ProgressColumn(
                "Spending score",
                min_value=0,
                max_value=100,
                format="%d",
            ),
            "Spending_Category": st.column_config.TextColumn("Segment"),
            "Recommended_Strategy": st.column_config.TextColumn("Idea to consider"),
        },
    )
    download = directory[display_columns].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download these records",
        data=download,
        file_name="filtered_mall_customers.csv",
        mime="text/csv",
        disabled=directory.empty,
        type="primary",
    )
    st.caption(f"Showing {len(directory):,} of {len(filtered):,} customers in the current filter.")
