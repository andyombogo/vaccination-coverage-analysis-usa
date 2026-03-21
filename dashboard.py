from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


DATA_FILE = Path(__file__).with_name("vaccination_data.csv")


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

    chart_col_1, chart_col_2 = st.columns((1.2, 1))
    with chart_col_1:
        st.subheader("Coverage trend")
        st.altair_chart(build_trend_chart(filtered), use_container_width=True)

    with chart_col_2:
        st.subheader("Highest-coverage geographies")
        st.altair_chart(build_geography_chart(filtered), use_container_width=True)

    st.subheader("Dimension snapshot")
    st.altair_chart(build_dimension_chart(filtered), use_container_width=True)

    summary_col, download_col = st.columns((1.2, 0.8))
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
            "Export the currently filtered slice to support reporting, quick QA, or downstream analysis."
        )
        st.download_button(
            "Download filtered CSV",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="vaccination_coverage_filtered.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.info(
            "Deployment note: the hosted app runs this dashboard, while the Spark scripts remain available for offline analysis."
        )


if __name__ == "__main__":
    main()
