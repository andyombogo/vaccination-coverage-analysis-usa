from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


DATA_FILE = Path(__file__).with_name("vaccination_data.csv")
REPO_URL = "https://github.com/andyombogo/vaccination-coverage-analysis-usa"


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE)
    df["Estimate (%)"] = pd.to_numeric(df["Estimate (%)"], errors="coerce")
    df["Sample Size"] = pd.to_numeric(df["Sample Size"], errors="coerce")
    df["Survey Year/Influenza Season"] = (
        df["Survey Year/Influenza Season"].astype(str).str.strip()
    )

    ci_bounds = df["95% CI (%)"].fillna("").str.extract(
        r"(?P<ci_low>\d+(?:\.\d+)?)\s*to\s*(?P<ci_high>\d+(?:\.\d+)?)"
    )
    df["CI Lower (%)"] = pd.to_numeric(ci_bounds["ci_low"], errors="coerce")
    df["CI Upper (%)"] = pd.to_numeric(ci_bounds["ci_high"], errors="coerce")
    df["Season Sort"] = (
        df["Survey Year/Influenza Season"].str.extract(r"(\d{4})")[0].fillna("0").astype(int)
    )

    return df.dropna(subset=["Estimate (%)", "Sample Size"]).copy()


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Explore the data")
    st.sidebar.caption(
        "Start broad, then narrow by vaccine, season, geography, and subgroup to surface coverage gaps."
    )

    vaccine_options = sorted(df["Vaccine"].dropna().unique().tolist())
    selected_vaccines = st.sidebar.multiselect(
        "Vaccine",
        vaccine_options,
        default=vaccine_options,
    )

    geography_type_options = sorted(df["Geography Type"].dropna().unique().tolist())
    selected_geography_types = st.sidebar.multiselect(
        "Geography type",
        geography_type_options,
        default=geography_type_options,
    )

    season_options = (
        df[["Survey Year/Influenza Season", "Season Sort"]]
        .drop_duplicates()
        .sort_values(["Season Sort", "Survey Year/Influenza Season"])
    )["Survey Year/Influenza Season"].tolist()
    selected_seasons = st.sidebar.multiselect(
        "Season",
        season_options,
        default=season_options,
    )

    dimension_type_options = sorted(df["Dimension Type"].dropna().unique().tolist())
    selected_dimension_types = st.sidebar.multiselect(
        "Dimension type",
        dimension_type_options,
        default=dimension_type_options,
    )

    filtered = df[
        df["Vaccine"].isin(selected_vaccines)
        & df["Geography Type"].isin(selected_geography_types)
        & df["Survey Year/Influenza Season"].isin(selected_seasons)
        & df["Dimension Type"].isin(selected_dimension_types)
    ].copy()

    geography_options = sorted(filtered["Geography"].dropna().unique().tolist())
    selected_geographies = st.sidebar.multiselect(
        "Geography",
        geography_options,
        default=[],
        help="Leave this empty to compare all selected geographies.",
    )

    if selected_geographies:
        filtered = filtered[filtered["Geography"].isin(selected_geographies)]

    return filtered


def format_pct(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:.1f}%"


def weighted_average(df: pd.DataFrame) -> float:
    sample_total = df["Sample Size"].sum()
    if sample_total == 0:
        return float("nan")
    return (df["Estimate (%)"] * df["Sample Size"]).sum() / sample_total


def top_geography_label(df: pd.DataFrame) -> str:
    summary = (
        df.groupby("Geography", as_index=False)["Estimate (%)"]
        .mean()
        .sort_values("Estimate (%)", ascending=False)
    )
    if summary.empty:
        return "N/A"
    top_row = summary.iloc[0]
    return f"{top_row['Geography']} ({top_row['Estimate (%)']:.1f}%)"


def latest_season(df: pd.DataFrame) -> str:
    season_table = (
        df[["Survey Year/Influenza Season", "Season Sort"]]
        .drop_duplicates()
        .sort_values(["Season Sort", "Survey Year/Influenza Season"])
    )
    if season_table.empty:
        return "N/A"
    return season_table.iloc[-1]["Survey Year/Influenza Season"]


def summarize_insights(df: pd.DataFrame) -> list[str]:
    insights: list[str] = []

    vaccine_summary = (
        df.groupby("Vaccine", as_index=False)["Estimate (%)"]
        .mean()
        .sort_values("Estimate (%)", ascending=False)
    )
    if not vaccine_summary.empty:
        best = vaccine_summary.iloc[0]
        worst = vaccine_summary.iloc[-1]
        insights.append(
            f"{best['Vaccine']} has the strongest average coverage in the current slice at {best['Estimate (%)']:.1f}%, while {worst['Vaccine']} trails at {worst['Estimate (%)']:.1f}%."
        )

    current_latest = latest_season(df)
    latest_df = df[df["Survey Year/Influenza Season"] == current_latest]
    if not latest_df.empty:
        latest_geo = (
            latest_df.groupby("Geography", as_index=False)["Estimate (%)"]
            .mean()
            .sort_values("Estimate (%)", ascending=False)
        )
        if not latest_geo.empty:
            leader = latest_geo.iloc[0]
            insights.append(
                f"In {current_latest}, {leader['Geography']} leads the selected geographies with an average coverage of {leader['Estimate (%)']:.1f}%."
            )

        gap_summary = []
        for dimension_type, group in latest_df.groupby("Dimension Type"):
            by_dimension = group.groupby("Dimension", as_index=False)["Estimate (%)"].mean()
            if len(by_dimension) < 2:
                continue
            spread = by_dimension["Estimate (%)"].max() - by_dimension["Estimate (%)"].min()
            gap_summary.append((dimension_type, spread))
        if gap_summary:
            widest = max(gap_summary, key=lambda item: item[1])
            insights.append(
                f"The widest subgroup gap in the latest season appears in {widest[0]}, with an average spread of {widest[1]:.1f} percentage points."
            )

    return insights


def build_trend_chart(df: pd.DataFrame) -> alt.Chart:
    trend = (
        df.groupby(["Season Sort", "Survey Year/Influenza Season", "Vaccine"], as_index=False)[
            "Estimate (%)"
        ]
        .mean()
        .sort_values(["Season Sort", "Survey Year/Influenza Season"])
    )
    return (
        alt.Chart(trend)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X(
                "Survey Year/Influenza Season:N",
                sort=trend["Survey Year/Influenza Season"].tolist(),
                title="Survey year / influenza season",
            ),
            y=alt.Y("Estimate (%):Q", title="Average coverage (%)"),
            color=alt.Color("Vaccine:N", title="Vaccine"),
            tooltip=[
                alt.Tooltip("Survey Year/Influenza Season:N", title="Season"),
                alt.Tooltip("Vaccine:N", title="Vaccine"),
                alt.Tooltip("Estimate (%):Q", title="Coverage", format=".1f"),
            ],
        )
        .properties(height=360)
    )


def build_geography_chart(df: pd.DataFrame) -> alt.Chart:
    by_geography = (
        df.groupby("Geography", as_index=False)["Estimate (%)"]
        .mean()
        .sort_values("Estimate (%)", ascending=False)
        .head(15)
    )
    return (
        alt.Chart(by_geography)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("Estimate (%):Q", title="Average coverage (%)"),
            y=alt.Y("Geography:N", sort="-x", title="Geography"),
            color=alt.Color(
                "Estimate (%):Q",
                scale=alt.Scale(scheme="tealblues"),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("Geography:N", title="Geography"),
                alt.Tooltip("Estimate (%):Q", title="Coverage", format=".1f"),
            ],
        )
        .properties(height=420)
    )


def build_vaccine_chart(df: pd.DataFrame) -> alt.Chart:
    by_vaccine = (
        df.groupby("Vaccine", as_index=False)["Estimate (%)"]
        .mean()
        .sort_values("Estimate (%)", ascending=False)
    )
    return (
        alt.Chart(by_vaccine)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("Vaccine:N", sort="-y", title="Vaccine"),
            y=alt.Y("Estimate (%):Q", title="Average coverage (%)"),
            color=alt.Color(
                "Estimate (%):Q",
                scale=alt.Scale(scheme="teals"),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("Vaccine:N", title="Vaccine"),
                alt.Tooltip("Estimate (%):Q", title="Coverage", format=".1f"),
            ],
        )
        .properties(height=320)
    )


def build_dimension_chart(df: pd.DataFrame) -> alt.Chart:
    by_dimension = (
        df.groupby("Dimension", as_index=False)["Estimate (%)"]
        .mean()
        .sort_values("Estimate (%)", ascending=False)
        .head(12)
    )
    return (
        alt.Chart(by_dimension)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("Dimension:N", sort="-y", title="Dimension"),
            y=alt.Y("Estimate (%):Q", title="Average coverage (%)"),
            color=alt.Color(
                "Estimate (%):Q",
                scale=alt.Scale(scheme="goldgreen"),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("Dimension:N", title="Dimension"),
                alt.Tooltip("Estimate (%):Q", title="Coverage", format=".1f"),
            ],
        )
        .properties(height=360)
    )


def build_gap_chart(df: pd.DataFrame) -> alt.Chart:
    rows = []
    for dimension_type, group in df.groupby("Dimension Type"):
        by_dimension = group.groupby("Dimension", as_index=False)["Estimate (%)"].mean()
        if len(by_dimension) < 2:
            continue
        rows.append(
            {
                "Dimension Type": dimension_type,
                "Coverage spread (%)": by_dimension["Estimate (%)"].max()
                - by_dimension["Estimate (%)"].min(),
            }
        )
    gap_df = pd.DataFrame(rows)
    if gap_df.empty:
        gap_df = pd.DataFrame(
            [{"Dimension Type": "Not enough subgroup data", "Coverage spread (%)": 0.0}]
        )
    return (
        alt.Chart(gap_df)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("Coverage spread (%):Q", title="Max minus min coverage"),
            y=alt.Y("Dimension Type:N", sort="-x", title="Dimension type"),
            color=alt.Color(
                "Coverage spread (%):Q",
                scale=alt.Scale(scheme="oranges"),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("Dimension Type:N", title="Dimension type"),
                alt.Tooltip("Coverage spread (%):Q", title="Spread", format=".1f"),
            ],
        )
        .properties(height=280)
    )


def build_confidence_interval_chart(
    df: pd.DataFrame,
    season: str,
    vaccine: str,
    geography: str,
    dimension_type: str,
) -> alt.Chart:
    subset = df[
        (df["Survey Year/Influenza Season"] == season)
        & (df["Vaccine"] == vaccine)
        & (df["Geography"] == geography)
        & (df["Dimension Type"] == dimension_type)
    ][["Dimension", "Estimate (%)", "CI Lower (%)", "CI Upper (%)"]].dropna()

    if subset.empty:
        subset = pd.DataFrame(
            {
                "Dimension": ["No confidence interval data"],
                "Estimate (%)": [0.0],
                "CI Lower (%)": [0.0],
                "CI Upper (%)": [0.0],
            }
        )

    subset = subset.sort_values("Estimate (%)", ascending=False)
    base = alt.Chart(subset).encode(
        y=alt.Y("Dimension:N", sort=subset["Dimension"].tolist(), title="Subgroup"),
        tooltip=[
            alt.Tooltip("Dimension:N", title="Subgroup"),
            alt.Tooltip("Estimate (%):Q", title="Estimate", format=".1f"),
            alt.Tooltip("CI Lower (%):Q", title="CI lower", format=".1f"),
            alt.Tooltip("CI Upper (%):Q", title="CI upper", format=".1f"),
        ],
    )
    rules = base.mark_rule(color="#557c86", strokeWidth=2).encode(
        x=alt.X("CI Lower (%):Q", title="Coverage estimate and confidence interval"),
        x2="CI Upper (%):Q",
    )
    points = base.mark_point(size=95, filled=True, color="#0f6d80").encode(
        x="Estimate (%):Q"
    )
    return (rules + points).properties(height=340)


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["Vaccine", "Geography", "Survey Year/Influenza Season"], as_index=False)
        .agg(
            average_coverage=("Estimate (%)", "mean"),
            average_sample_size=("Sample Size", "mean"),
            rows=("Estimate (%)", "size"),
        )
        .sort_values("average_coverage", ascending=False)
    )
    summary["average_coverage"] = summary["average_coverage"].round(1)
    summary["average_sample_size"] = summary["average_sample_size"].round(0).astype(int)
    return summary


def render_header(df: pd.DataFrame) -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(140, 190, 201, 0.16), transparent 40%),
                    linear-gradient(180deg, #f6fbfc 0%, #ffffff 100%);
            }
            .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
            }
            .hero {
                padding: 1.5rem 1.6rem;
                border: 1px solid rgba(39, 94, 110, 0.14);
                border-radius: 22px;
                background: linear-gradient(135deg, rgba(9, 91, 110, 0.08), rgba(244, 196, 48, 0.12));
                margin-bottom: 1.25rem;
            }
            .hero h1 {
                margin: 0;
                color: #0f3948;
                font-size: 2.4rem;
            }
            .hero p {
                margin: 0.6rem 0 0;
                max-width: 60rem;
                color: #34505b;
                font-size: 1rem;
            }
            .insight-card {
                padding: 1rem 1.1rem;
                border-radius: 18px;
                border: 1px solid rgba(15, 57, 72, 0.10);
                background: rgba(255, 255, 255, 0.88);
                box-shadow: 0 16px 40px rgba(15, 57, 72, 0.06);
                min-height: 132px;
            }
            .insight-card h3 {
                margin: 0 0 0.55rem;
                color: #0f3948;
                font-size: 1rem;
            }
            .insight-card p {
                margin: 0;
                color: #35545d;
                line-height: 1.5;
            }
            [data-testid="stMetricValue"] {
                color: #0f3948;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="hero">
            <h1>US Maternal Vaccination Coverage Dashboard</h1>
            <p>
                Explore CDC vaccination coverage estimates for pregnant women across vaccines,
                geographies, seasons, and demographic dimensions. This dashboard is optimized for
                lightweight cloud deployment and quick policy-oriented exploration.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        f"{len(df):,} cleaned records loaded from the CDC pregnancy vaccination coverage dataset."
    )


def main() -> None:
    st.set_page_config(
        page_title="Vaccination Coverage Dashboard",
        layout="wide",
    )

    data = load_data()
    render_header(data)
    filtered = apply_filters(data)

    if filtered.empty:
        st.warning("No rows match the current filters. Adjust the sidebar selections to continue.")
        return

    metric_cols = st.columns(4)
    metric_cols[0].metric("Records", f"{len(filtered):,}")
    metric_cols[1].metric("Average coverage", format_pct(filtered["Estimate (%)"].mean()))
    metric_cols[2].metric("Weighted coverage", format_pct(weighted_average(filtered)))
    metric_cols[3].metric("Top geography", top_geography_label(filtered))

    tabs = st.tabs(["Overview", "Geographies", "Equity Lens", "Data and Deployment"])

    with tabs[0]:
        chart_col_1, chart_col_2 = st.columns((1.2, 1))
        with chart_col_1:
            st.subheader("Coverage trend")
            st.altair_chart(build_trend_chart(filtered), use_container_width=True)

        with chart_col_2:
            st.subheader("Vaccine comparison")
            st.altair_chart(build_vaccine_chart(filtered), use_container_width=True)

        st.subheader("Key takeaways")
        insight_columns = st.columns(3)
        insights = summarize_insights(filtered)
        for index, insight in enumerate(insights[:3]):
            insight_columns[index].markdown(
                f"""
                <div class="insight-card">
                    <h3>Insight {index + 1}</h3>
                    <p>{insight}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with tabs[1]:
        geo_chart_col, geo_detail_col = st.columns((1.15, 0.85))
        with geo_chart_col:
            st.subheader("Highest-coverage geographies")
            st.altair_chart(build_geography_chart(filtered), use_container_width=True)

        with geo_detail_col:
            st.subheader("Current slice")
            st.write(
                "Use the sidebar filters to move from national patterns to individual states and subgroup slices."
            )
            st.metric("Latest season in view", latest_season(filtered))
            st.metric("Unique geographies", f"{filtered['Geography'].nunique():,}")
            st.metric("Unique subgroup values", f"{filtered['Dimension'].nunique():,}")

    with tabs[2]:
        st.subheader("Subgroup variation")
        gap_col, dimension_col = st.columns((0.9, 1.1))
        with gap_col:
            st.altair_chart(build_gap_chart(filtered), use_container_width=True)
        with dimension_col:
            st.altair_chart(build_dimension_chart(filtered), use_container_width=True)

        st.subheader("Confidence intervals for one slice")
        selection_cols = st.columns(4)
        ci_seasons = (
            filtered[["Survey Year/Influenza Season", "Season Sort"]]
            .drop_duplicates()
            .sort_values(["Season Sort", "Survey Year/Influenza Season"])
        )["Survey Year/Influenza Season"].tolist()
        ci_vaccines = sorted(filtered["Vaccine"].dropna().unique().tolist())
        ci_geographies = sorted(filtered["Geography"].dropna().unique().tolist())
        ci_dimension_types = sorted(filtered["Dimension Type"].dropna().unique().tolist())

        selected_ci_season = selection_cols[0].selectbox(
            "Season", ci_seasons, index=max(len(ci_seasons) - 1, 0)
        )
        selected_ci_vaccine = selection_cols[1].selectbox("Vaccine", ci_vaccines, index=0)
        selected_ci_geography = selection_cols[2].selectbox("Geography", ci_geographies, index=0)
        selected_ci_dimension_type = selection_cols[3].selectbox(
            "Dimension type", ci_dimension_types, index=0
        )

        st.altair_chart(
            build_confidence_interval_chart(
                filtered,
                selected_ci_season,
                selected_ci_vaccine,
                selected_ci_geography,
                selected_ci_dimension_type,
            ),
            use_container_width=True,
        )

    with tabs[3]:
        summary_col, download_col = st.columns((1.15, 0.85))
        with summary_col:
            st.subheader("Filtered summary")
            st.dataframe(
                build_summary_table(filtered),
                use_container_width=True,
                hide_index=True,
            )

        with download_col:
            st.subheader("Take the data")
            st.write(
                "Export the current filtered slice for reporting, quality checks, or downstream analysis."
            )
            st.download_button(
                "Download filtered CSV",
                data=filtered.to_csv(index=False).encode("utf-8"),
                file_name="vaccination_coverage_filtered.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.info(
                "The deployed app is lightweight by design. Spark scripts remain in the repo for offline analysis."
            )

        st.subheader("Deploy this project")
        st.markdown(
            f"- Repo: [{REPO_URL}]({REPO_URL})\n"
            "- Platform: Render Blueprint or any Docker-compatible host\n"
            "- Start command: `streamlit run dashboard.py --server.address 0.0.0.0 --server.port $PORT`"
        )
        st.caption(
            "A public Render URL is generated only after the service is created in a Render account and the first deploy succeeds."
        )


if __name__ == "__main__":
    main()
