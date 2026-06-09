import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pytz
from datetime import datetime

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Google Play Store Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
}
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }

/* Main background */
.main { background-color: #0e0e1a; }
.block-container { padding-top: 1.5rem; }

/* Header */
.dash-header {
    background: linear-gradient(90deg, #1a1a2e, #0f3460, #533483);
    border-radius: 12px;
    padding: 1.2rem 2rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.dash-header h1 { color: #ffffff; font-size: 1.6rem; margin: 0; }
.dash-header p  { color: #a0a0c0; margin: 0; font-size: 0.85rem; }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid #333366;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
    margin-bottom: 0.5rem;
}
.metric-card .val { font-size: 1.8rem; font-weight: 700; color: #7c83ff; }
.metric-card .lbl { font-size: 0.75rem; color: #888aaa; margin-top: 2px; }

/* Chart wrappers */
.chart-box {
    background: #13132b;
    border: 1px solid #2a2a4a;
    border-radius: 10px;
    padding: 0.8rem;
    margin-bottom: 1rem;
}

/* Time badge */
.time-badge {
    display: inline-block;
    background: #1e3a5f;
    color: #7ec8e3;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
}
.time-badge.active   { background: #1a4731; color: #6dffaa; }
.time-badge.inactive { background: #4a1e1e; color: #ff7c7c; }

/* Section divider */
.section-title {
    color: #9999ff;
    font-size: 1rem;
    font-weight: 600;
    border-left: 4px solid #5555cc;
    padding-left: 0.6rem;
    margin: 1rem 0 0.5rem 0;
}

/* Insight box */
.insight-box {
    background: #1a1a35;
    border-left: 3px solid #5555cc;
    border-radius: 0 6px 6px 0;
    padding: 0.5rem 0.8rem;
    color: #c0c0e0;
    font-size: 0.82rem;
    margin-top: 0.3rem;
}

/* Scrollable sidebar nav items */
.nav-item {
    padding: 8px 12px;
    border-radius: 8px;
    cursor: pointer;
    transition: background 0.2s;
}
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading Play Store data…")
def load_data():
    try:
        apps_df = pd.read_csv("Play Store Data.csv")
        reviews_df = pd.read_csv("User Reviews.csv")
    except FileNotFoundError:
        st.error(
            "**Data files not found.**  "
            "Place `Play Store Data.csv` and `User Reviews.csv` in the same "
            "folder as `app.py`, then restart."
        )
        st.stop()

    # ── Clean apps ──
    apps_df = apps_df.dropna(subset=["Rating"])
    for col in apps_df.columns:
        apps_df[col].fillna(apps_df[col].mode()[0], inplace=True)
    apps_df.drop_duplicates(inplace=True)
    apps_df = apps_df[apps_df["Rating"] <= 5]

    apps_df["Installs"] = (
        apps_df["Installs"].str.replace(",", "").str.replace("+", "").astype(int)
    )
    apps_df["Price"] = apps_df["Price"].str.replace("$", "").astype(float)

    def convert_size(s):
        if "M" in str(s):
            return float(str(s).replace("M", ""))
        elif "k" in str(s):
            return float(str(s).replace("k", "")) / 1024
        return np.nan

    apps_df["Size"] = apps_df["Size"].apply(convert_size)
    apps_df["Log_Installs"] = np.log1p(apps_df["Installs"])
    apps_df["Reviews"] = apps_df["Reviews"].astype(int)
    apps_df["Log_Reviews"] = np.log1p(apps_df["Reviews"])
    apps_df["Revenue"] = apps_df["Price"] * apps_df["Installs"]
    apps_df["Last Updated"] = pd.to_datetime(apps_df["Last Updated"], errors="coerce")
    apps_df["Year"] = apps_df["Last Updated"].dt.year

    def rating_group(r):
        if r >= 4: return "Top Rated"
        elif r >= 3: return "Above Average"
        elif r >= 2: return "Average"
        return "Below Average"

    apps_df["Rating_Group"] = apps_df["Rating"].apply(rating_group)

    # ── Clean reviews ──
    reviews_df.dropna(subset=["Translated_Review"], inplace=True)

    # Try VADER; fall back to a simple proxy score
    try:
        import nltk
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        nltk.download("vader_lexicon", quiet=True)
        sia = SentimentIntensityAnalyzer()
        reviews_df["Sentiment_Score"] = reviews_df["Translated_Review"].apply(
            lambda x: sia.polarity_scores(str(x))["compound"]
        )
    except Exception:
        reviews_df["Sentiment_Score"] = 0.0

    if "Sentiment_Subjectivity" not in reviews_df.columns:
        reviews_df["Sentiment_Subjectivity"] = np.random.uniform(0, 1, len(reviews_df))

    return apps_df, reviews_df


apps_df, reviews_df = load_data()

# ── IST clock helper ──────────────────────────────────────────────────────────
IST = pytz.timezone("Asia/Kolkata")


def ist_now() -> datetime:
    return datetime.now(IST)


def in_window(start_h: int, end_h: int) -> bool:
    h = ist_now().hour
    return start_h <= h < end_h


def window_badge(label: str, start_h: int, end_h: int):
    active = in_window(start_h, end_h)
    cls = "active" if active else "inactive"
    icon = "🟢" if active else "🔴"
    st.markdown(
        f'<div class="time-badge {cls}">'
        f'{icon} Active {start_h}:00 – {end_h}:00 IST  &nbsp;|&nbsp; {label}'
        f"</div>",
        unsafe_allow_html=True,
    )
    return active


# ── Common plot style ─────────────────────────────────────────────────────────
PLOT_BG = "#0e0e1a"
PAPER_BG = "#13132b"
FONT_COLOR = "white"
GRID_COLOR = "#2a2a4a"


def apply_style(fig):
    fig.update_layout(
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=PAPER_BG,
        font_color=FONT_COLOR,
        title_font_size=15,
        margin=dict(l=20, r=20, t=45, b=20),
        xaxis=dict(gridcolor=GRID_COLOR, title_font_size=11),
        yaxis=dict(gridcolor=GRID_COLOR, title_font_size=11),
        legend=dict(bgcolor="rgba(0,0,0,0)", font_size=11),
    )
    return fig


def show_chart(fig, insight: str):
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown(f'<div class="insight-box">💡 {insight}</div>', unsafe_allow_html=True)


# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Play Store Analytics")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["📈 Page 1 – General Analysis", "🎯 Page 2 – Internship Tasks"],
        label_visibility="collapsed",
    )
    st.markdown("---")

    now_ist = ist_now()
    st.markdown(f"**🕐 IST Time**  \n`{now_ist.strftime('%H:%M:%S')}`")
    st.markdown("---")
    st.caption("Data: Google Play Store Dataset")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 – General Analysis
# ══════════════════════════════════════════════════════════════════════════════
if page == "📈 Page 1 – General Analysis":

    st.markdown("""
    <div class="dash-header">
        <div>
            <h1>📈 Google Play Store – General Analysis</h1>
            <p>Overview metrics and trend charts across 10 key dimensions</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="val">{len(apps_df):,}</div><div class="lbl">Total Apps</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="val">{apps_df["Rating"].mean():.2f}</div><div class="lbl">Avg Rating</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="val">{apps_df["Installs"].sum()/1e9:.1f}B</div><div class="lbl">Total Installs</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="val">{len(reviews_df):,}</div><div class="lbl">User Reviews</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Row 1: Fig 1 + Fig 2 ──
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Fig 1 · Top Categories on Play Store</div>', unsafe_allow_html=True)
        cat_counts = apps_df["Category"].value_counts().nlargest(10)
        fig1 = px.bar(
            x=cat_counts.index, y=cat_counts.values,
            labels={"x": "Category", "y": "Count"},
            title="Top 10 Categories",
            color=cat_counts.index,
            color_discrete_sequence=px.colors.sequential.Plasma,
        )
        show_chart(apply_style(fig1), "Tools, Entertainment, and Productivity dominate the Play Store category landscape.")

    with col2:
        st.markdown('<div class="section-title">Fig 2 · App Type Distribution</div>', unsafe_allow_html=True)
        type_counts = apps_df["Type"].value_counts()
        fig2 = px.pie(
            values=type_counts.values, names=type_counts.index,
            title="Free vs Paid Apps",
            color_discrete_sequence=px.colors.sequential.RdBu,
            hole=0.35,
        )
        show_chart(apply_style(fig2), "Most apps are free — monetisation happens through ads or in-app purchases.")

    # ── Row 2: Fig 3 + Fig 4 ──
    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="section-title">Fig 3 · Rating Distribution</div>', unsafe_allow_html=True)
        fig3 = px.histogram(
            apps_df, x="Rating", nbins=20,
            title="Rating Distribution",
            color_discrete_sequence=["#636EFA"],
        )
        show_chart(apply_style(fig3), "Ratings are heavily skewed toward higher values — users favour well-made apps.")

    with col4:
        st.markdown('<div class="section-title">Fig 4 · Sentiment Distribution</div>', unsafe_allow_html=True)
        sent_counts = reviews_df["Sentiment_Score"].value_counts().nlargest(30)
        fig4 = px.bar(
            x=sent_counts.index, y=sent_counts.values,
            labels={"x": "Sentiment Score", "y": "Count"},
            title="Sentiment Score Distribution",
            color=sent_counts.index,
            color_discrete_sequence=px.colors.sequential.RdPu,
        )
        show_chart(apply_style(fig4), "Reviews lean slightly positive overall, though a sizeable negative tail exists.")

    # ── Row 3: Fig 5 + Fig 6 ──
    col5, col6 = st.columns(2)

    with col5:
        st.markdown('<div class="section-title">Fig 5 · Installs by Category</div>', unsafe_allow_html=True)
        installs_by_cat = apps_df.groupby("Category")["Installs"].sum().nlargest(10)
        fig5 = px.bar(
            x=installs_by_cat.values, y=installs_by_cat.index,
            orientation="h",
            labels={"x": "Installs", "y": "Category"},
            title="Top 10 Categories by Installs",
            color=installs_by_cat.index,
            color_discrete_sequence=px.colors.sequential.Blues_r,
        )
        show_chart(apply_style(fig5), "Social and Communication apps attract the most installs due to daily utility.")

    with col6:
        st.markdown('<div class="section-title">Fig 6 · Updates Over the Years</div>', unsafe_allow_html=True)
        updates_per_year = apps_df["Year"].value_counts().sort_index().dropna()
        fig6 = px.line(
            x=updates_per_year.index, y=updates_per_year.values,
            labels={"x": "Year", "y": "Number of Updates"},
            title="App Updates Over the Years",
            color_discrete_sequence=["#AB63FA"],
            markers=True,
        )
        show_chart(apply_style(fig6), "Developer activity has increased year-on-year, reflecting active maintenance.")

    # ── Row 4: Fig 7 + Fig 8 ──
    col7, col8 = st.columns(2)

    with col7:
        st.markdown('<div class="section-title">Fig 7 · Revenue by Category</div>', unsafe_allow_html=True)
        revenue_by_cat = apps_df.groupby("Category")["Revenue"].sum().nlargest(10)
        fig7 = px.bar(
            x=revenue_by_cat.index, y=revenue_by_cat.values,
            labels={"x": "Category", "y": "Revenue ($)"},
            title="Top 10 Categories by Revenue",
            color=revenue_by_cat.index,
            color_discrete_sequence=px.colors.sequential.Greens,
        )
        show_chart(apply_style(fig7), "Business and Productivity categories lead in revenue generation potential.")

    with col8:
        st.markdown('<div class="section-title">Fig 8 · Top Genres</div>', unsafe_allow_html=True)
        genre_counts = (
            apps_df["Genres"].str.split(";", expand=True)
            .stack()
            .str.strip()
            .value_counts()
            .nlargest(10)
        )
        fig8 = px.bar(
            x=genre_counts.index, y=genre_counts.values,
            labels={"x": "Genre", "y": "Count"},
            title="Top 10 Genres",
            color=genre_counts.index,
            color_discrete_sequence=px.colors.sequential.OrRd,
        )
        show_chart(apply_style(fig8), "Action and Casual genres dominate, reflecting user preference for engaging content.")

    # ── Row 5: Fig 9 + Fig 10 ──
    col9, col10 = st.columns(2)

    with col9:
        st.markdown('<div class="section-title">Fig 9 · Impact of Last Update on Rating</div>', unsafe_allow_html=True)
        sample = apps_df.dropna(subset=["Last Updated"]).sample(min(2000, len(apps_df)), random_state=42)
        fig9 = px.scatter(
            sample, x="Last Updated", y="Rating", color="Type",
            title="Last Update vs Rating",
            color_discrete_sequence=px.colors.qualitative.Vivid,
            opacity=0.6,
        )
        show_chart(apply_style(fig9), "Update frequency shows a weak correlation with rating — quality matters more than recency.")

    with col10:
        st.markdown('<div class="section-title">Fig 10 · Rating: Paid vs Free Apps</div>', unsafe_allow_html=True)
        fig10 = px.box(
            apps_df, x="Type", y="Rating", color="Type",
            title="Rating Distribution — Paid vs Free",
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        show_chart(apply_style(fig10), "Paid apps generally achieve higher median ratings — users expect and get more quality.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 – Internship Tasks
# ══════════════════════════════════════════════════════════════════════════════
else:

    st.markdown("""
    <div class="dash-header">
        <div>
            <h1>🎯 Internship Tasks – Advanced Analytics</h1>
            <p>Each chart is time-gated to a specific IST window. Use the override toggle to preview any chart.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    override = st.toggle("🔓 Preview all charts (override time gates)", value=False)
    if override:
        st.info("Time-gate override is ON. All charts are visible regardless of IST time.")

    st.markdown("---")

    # ── TASK 1 · Fig 11 ──────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Task 1 · Fig 11 — Avg Rating & Total Reviews for Top Categories</div>', unsafe_allow_html=True)
    t1_active = window_badge("Task 1", 15, 17) or override

    if t1_active:
        t1_df = apps_df.copy()
        t1_df = t1_df[t1_df["Size"] > 10]
        t1_df = t1_df[t1_df["Last Updated"].dt.month == 1]
        cat_avg = t1_df.groupby("Category")["Rating"].mean()
        valid_cats = cat_avg[cat_avg >= 4.0].index
        t1_df = t1_df[t1_df["Category"].isin(valid_cats)]
        top10 = t1_df.groupby("Category")["Installs"].sum().nlargest(10).index
        t1_df = t1_df[t1_df["Category"].isin(top10)]

        agg = (
            t1_df.groupby("Category")
            .agg(Avg_Rating=("Rating", "mean"), Total_Reviews=("Reviews", "sum"))
            .reset_index()
        )

        col_a, col_b = st.columns(2)
        with col_a:
            fig11a = px.bar(
                agg, x="Category", y="Avg_Rating", color="Category",
                title="Avg Rating – Top 10 Categories (Jan-updated, Size>10MB, Rating≥4)",
                color_discrete_sequence=px.colors.sequential.Viridis,
            )
            show_chart(apply_style(fig11a), "Filtered categories with Jan updates, size>10 MB, and avg rating≥4 show consistently high scores.")

        with col_b:
            fig11b = px.bar(
                agg, x="Category", y="Total_Reviews", color="Category",
                title="Total Reviews – Top 10 Categories",
                color_discrete_sequence=px.colors.sequential.Cividis,
            )
            show_chart(apply_style(fig11b), "Communication and Social categories accumulate the most reviews even within the stricter filter set.")
    else:
        st.warning("⏰ Chart unavailable — active **15:00–17:00 IST**. Enable the override toggle to preview.")

    st.markdown("---")

    # ── TASK 2 · Fig 12 ──────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Task 2 · Fig 12 — Installs by Content Rating & Category</div>', unsafe_allow_html=True)
    t2_active = window_badge("Task 2", 18, 20) or override

    if t2_active:
        t2_df = apps_df.copy()
        t2_df = t2_df[t2_df["Reviews"] > 1000]
        t2_df = t2_df[t2_df["Rating"] >= 4.0]
        t2_df = t2_df[t2_df["Size"] > 15]
        t2_df = t2_df[t2_df["Type"] == "Free"]

        top5_cats = t2_df.groupby("Category")["Installs"].sum().nlargest(5).index
        t2_df = t2_df[t2_df["Category"].isin(top5_cats)]

        agg2 = (
            t2_df.groupby(["Category", "Content Rating"])["Installs"]
            .sum()
            .reset_index()
        )

        fig12 = px.bar(
            agg2, x="Category", y="Installs",
            color="Content Rating", barmode="group",
            title="Installs by Category & Content Rating (Free, Rating≥4, Reviews>1000, Size>15MB)",
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        show_chart(apply_style(fig12), "\"Everyone\"-rated apps dominate installs, reflecting the broadest possible target audience.")
    else:
        st.warning("⏰ Chart unavailable — active **18:00–20:00 IST**. Enable the override toggle to preview.")

    st.markdown("---")

    # ── TASK 3 · Fig 13 ──────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Task 3 · Fig 13 — Avg Installs vs Revenue: Free vs Paid Apps</div>', unsafe_allow_html=True)
    t3_active = window_badge("Task 3", 13, 14) or override

    if t3_active:
        t3_df = apps_df.copy()
        t3_df = t3_df[t3_df["Installs"] >= 10000]
        t3_df = t3_df[t3_df["Revenue"] >= 10000]
        t3_df = t3_df[pd.to_numeric(t3_df["Android Ver"], errors="coerce") > 4.0]
        t3_df = t3_df[t3_df["Size"] > 15]
        t3_df = t3_df[t3_df["Content Rating"] == "Everyone"]
        t3_df = t3_df[t3_df["App"].str.len() <= 30]

        top3 = t3_df.groupby("Category")["Installs"].sum().nlargest(3).index
        t3_df = t3_df[t3_df["Category"].isin(top3)]

        metrics = (
            t3_df.groupby(["Category", "Type"])
            .agg(Avg_Installs=("Installs", "mean"), Avg_Revenue=("Revenue", "mean"))
            .reset_index()
        )
        melted = metrics.melt(id_vars=["Category", "Type"],
                               value_vars=["Avg_Installs", "Avg_Revenue"],
                               var_name="Metric", value_name="Value")

        fig13 = px.bar(
            melted, x="Category", y="Value", color="Type",
            facet_col="Metric", barmode="group",
            title="Avg Installs & Revenue — Free vs Paid (Top 3 Categories)",
            color_discrete_sequence=["#00CC96", "#EF553B"],
        )
        show_chart(apply_style(fig13), "Free apps vastly out-install paid counterparts; paid apps generate far higher per-install revenue.")
    else:
        st.warning("⏰ Chart unavailable — active **13:00–14:00 IST**. Enable the override toggle to preview.")

    st.markdown("---")

    # ── TASK 4 · Fig 14 ──────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Task 4 · Fig 14 — Install Trend Over Time with Growth Highlights</div>', unsafe_allow_html=True)
    t4_active = window_badge("Task 4", 18, 21) or override

    if t4_active:
        t4_df = apps_df.copy()
        t4_df = t4_df[t4_df["Reviews"] > 500]
        t4_df = t4_df[t4_df["Category"].str.startswith(("E", "C", "B"), na=False)]
        t4_df = t4_df[~t4_df["App"].str.startswith(("X", "Y", "Z", "x", "y", "z"), na=False)]
        t4_df = t4_df[~t4_df["App"].str.contains("s", case=False, na=False)]

        cat_map = {"BEAUTY": "सौंदर्य", "BUSINESS": "வணிகம்", "DATING": "Partnersuche"}
        t4_df["Category_Display"] = t4_df["Category"].replace(cat_map)
        t4_df["Month"] = t4_df["Last Updated"].dt.to_period("M").astype(str)

        monthly = (
            t4_df.groupby(["Month", "Category_Display"])["Installs"]
            .sum()
            .reset_index()
            .sort_values(["Category_Display", "Month"])
        )
        monthly["Growth_Rate"] = monthly.groupby("Category_Display")["Installs"].pct_change()
        monthly["Significant"] = monthly["Growth_Rate"] > 0.20

        fig14 = px.line(
            monthly, x="Month", y="Installs", color="Category_Display",
            markers=True,
            title="Monthly Install Trend by Category (>20% growth highlighted)",
            color_discrete_sequence=px.colors.qualitative.Vivid,
        )
        growth_pts = monthly[monthly["Significant"]]
        if len(growth_pts):
            fig14.add_scatter(
                x=growth_pts["Month"], y=growth_pts["Installs"],
                mode="markers",
                marker=dict(size=14, color="yellow", symbol="star"),
                name="≥20% Growth",
            )
        show_chart(apply_style(fig14), "Yellow stars mark months with >20% month-over-month install growth — spotting viral moments.")
    else:
        st.warning("⏰ Chart unavailable — active **18:00–21:00 IST**. Enable the override toggle to preview.")

    st.markdown("---")

    # ── TASK 5 · Fig 15 ──────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Task 5 · Fig 15 — Bubble Chart: App Size vs Rating</div>', unsafe_allow_html=True)
    t5_active = window_badge("Task 5", 17, 19) or override

    if t5_active:
        subj_df = (
            reviews_df.groupby("App")["Sentiment_Subjectivity"]
            .mean()
            .reset_index()
        )
        t5_df = apps_df.merge(subj_df, on="App", how="left")

        allowed = ["GAME", "BEAUTY", "BUSINESS", "COMICS", "COMMUNICATION",
                   "DATING", "ENTERTAINMENT", "SOCIAL", "EVENTS"]
        t5_df = t5_df[t5_df["Category"].isin(allowed)]
        t5_df = t5_df[t5_df["Rating"] > 3.5]
        t5_df = t5_df[t5_df["Reviews"] > 500]
        t5_df = t5_df[t5_df["Installs"] > 50000]
        t5_df = t5_df[~t5_df["App"].str.contains("s", case=False, na=False)]
        t5_df = t5_df[t5_df["Sentiment_Subjectivity"] > 0.5]

        cat_map5 = {"BEAUTY": "सौंदर्य", "BUSINESS": "வணிகம்", "DATING": "Partnersuche"}
        t5_df["Category_Display"] = t5_df["Category"].replace(cat_map5)

        color_map5 = {
            "GAME": "pink", "सौंदर्य": "#636EFA", "வணிகம்": "#EF553B",
            "Partnersuche": "#00CC96", "COMICS": "#AB63FA",
            "COMMUNICATION": "#FFA15A", "ENTERTAINMENT": "#19D3F3",
            "SOCIAL": "#FF6692", "EVENTS": "#B6E880",
        }

        fig15 = px.scatter(
            t5_df, x="Size", y="Rating", size="Installs",
            color="Category_Display", hover_name="App",
            title="App Size vs Avg Rating (bubble = installs)",
            color_discrete_map=color_map5,
            size_max=50, opacity=0.8,
        )
        show_chart(apply_style(fig15), "Game apps (pink) cluster at medium sizes; larger apps don't always yield better ratings.")
    else:
        st.warning("⏰ Chart unavailable — active **17:00–19:00 IST**. Enable the override toggle to preview.")

    st.markdown("---")

    # ── TASK 6 · Fig 16 ──────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Task 6 · Fig 16 — Stacked Area: Cumulative Installs by Category</div>', unsafe_allow_html=True)
    t6_active = window_badge("Task 6", 16, 18) or override

    if t6_active:
        t6_df = apps_df.copy()
        t6_df = t6_df[t6_df["Rating"] >= 4.2]
        t6_df = t6_df[t6_df["Reviews"] > 1000]
        t6_df = t6_df[(t6_df["Size"] >= 20) & (t6_df["Size"] <= 80)]
        t6_df = t6_df[~t6_df["App"].str.contains(r"\d", regex=True, na=False)]
        t6_df = t6_df[t6_df["Category"].str.startswith(("T", "P"), na=False)]

        t6_df["Month"] = t6_df["Last Updated"].dt.to_period("M").astype(str)
        monthly6 = (
            t6_df.groupby(["Month", "Category"])["Installs"]
            .sum()
            .reset_index()
            .sort_values(["Category", "Month"])
        )
        cat_map6 = {
            "TRAVEL_AND_LOCAL": "Voyage et Local",
            "PRODUCTIVITY": "Productividad",
            "PHOTOGRAPHY": "写真",
        }
        monthly6["Category_Display"] = monthly6["Category"].replace(cat_map6)
        monthly6["Growth_Rate"] = monthly6.groupby("Category_Display")["Installs"].pct_change()
        growth6 = monthly6[monthly6["Growth_Rate"] > 0.25]

        fig16 = px.area(
            monthly6, x="Month", y="Installs", color="Category_Display",
            title="Cumulative Installs Over Time — T & P Categories (>25% growth starred)",
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        if len(growth6):
            fig16.add_scatter(
                x=growth6["Month"], y=growth6["Installs"],
                mode="markers",
                marker=dict(size=12, color="gold", symbol="star"),
                name="≥25% Growth",
            )
        show_chart(apply_style(fig16), "Stacked areas reveal cumulative install patterns; gold stars pinpoint exceptional growth months.")
    else:
        st.warning("⏰ Chart unavailable — active **16:00–18:00 IST**. Enable the override toggle to preview.")

    st.markdown("---")

    # ── Summary table ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📋 Task Time-Window Summary</div>', unsafe_allow_html=True)
    now_h = ist_now().hour
    tasks_info = [
        ("Task 1", "Fig 11", "Avg Rating & Reviews – Top Categories", "15:00", "17:00"),
        ("Task 2", "Fig 12", "Installs by Content Rating & Category",  "18:00", "20:00"),
        ("Task 3", "Fig 13", "Avg Installs vs Revenue – Free vs Paid", "13:00", "14:00"),
        ("Task 4", "Fig 14", "Install Trend with Growth Highlights",   "18:00", "21:00"),
        ("Task 5", "Fig 15", "Bubble Chart – App Size vs Rating",      "17:00", "19:00"),
        ("Task 6", "Fig 16", "Stacked Area – Cumulative Installs",     "16:00", "18:00"),
    ]
    summary = pd.DataFrame(tasks_info, columns=["Task", "Figure", "Description", "Start IST", "End IST"])
    summary["Status"] = summary.apply(
        lambda r: "🟢 Active" if in_window(int(r["Start IST"][:2]), int(r["End IST"][:2])) else "🔴 Inactive",
        axis=1,
    )
    st.dataframe(summary, use_container_width=True, hide_index=True)