import math
import unittest
from unittest.mock import patch

import pandas as pd

import dashboard


def sample_records() -> list[dict]:
    return [
        {
            "Vaccine": "Tdap",
            "Geography Type": "States",
            "Geography": "Ohio",
            "Survey Year/Influenza Season": "2022",
            "Dimension Type": "Race and Ethnicity",
            "Dimension": "White, Non-Hispanic",
            "Estimate (%)": 70.0,
            "95% CI (%)": "66.0 to 74.0",
            "Sample Size": 100,
        },
        {
            "Vaccine": "Tdap",
            "Geography Type": "States",
            "Geography": "Ohio",
            "Survey Year/Influenza Season": "2022",
            "Dimension Type": "Race and Ethnicity",
            "Dimension": "Black, Non-Hispanic",
            "Estimate (%)": 50.0,
            "95% CI (%)": "45.0 to 55.0",
            "Sample Size": 80,
        },
        {
            "Vaccine": "Influenza",
            "Geography Type": "States",
            "Geography": "Texas",
            "Survey Year/Influenza Season": "2021",
            "Dimension Type": "Age",
            "Dimension": "18-24",
            "Estimate (%)": 45.0,
            "95% CI (%)": "40.0 to 50.0",
            "Sample Size": 60,
        },
        {
            "Vaccine": "Influenza",
            "Geography Type": "States",
            "Geography": "Texas",
            "Survey Year/Influenza Season": "2021",
            "Dimension Type": "Age",
            "Dimension": "25-34",
            "Estimate (%)": 55.0,
            "95% CI (%)": "50.0 to 60.0",
            "Sample Size": 40,
        },
    ]


class DashboardHelpersTest(unittest.TestCase):
    def setUp(self) -> None:
        self.df = pd.DataFrame(sample_records())
        self.df["CI Lower (%)"] = [66.0, 45.0, 40.0, 50.0]
        self.df["CI Upper (%)"] = [74.0, 55.0, 50.0, 60.0]
        self.df["Season Sort"] = [2022, 2022, 2021, 2021]

    def test_format_pct_handles_missing_values(self) -> None:
        self.assertEqual(dashboard.format_pct(61.234), "61.2%")
        self.assertEqual(dashboard.format_pct(float("nan")), "N/A")

    def test_weighted_average_uses_sample_size_weights(self) -> None:
        result = dashboard.weighted_average(self.df)
        expected = ((70 * 100) + (50 * 80) + (45 * 60) + (55 * 40)) / (100 + 80 + 60 + 40)
        self.assertAlmostEqual(result, expected)

    def test_top_geography_label_returns_best_average(self) -> None:
        self.assertEqual(dashboard.top_geography_label(self.df), "Ohio (60.0%)")

    def test_latest_season_picks_highest_sort_order(self) -> None:
        self.assertEqual(dashboard.latest_season(self.df), "2022")

    def test_summarize_insights_returns_human_readable_findings(self) -> None:
        insights = dashboard.summarize_insights(self.df)
        self.assertGreaterEqual(len(insights), 2)
        self.assertIn("Tdap", insights[0])
        self.assertTrue(any("2022" in insight for insight in insights))

    def test_build_summary_table_rounds_and_sorts(self) -> None:
        summary = dashboard.build_summary_table(self.df)
        self.assertEqual(
            list(summary.columns),
            ["Vaccine", "Geography", "Survey Year/Influenza Season", "average_coverage", "average_sample_size", "rows"],
        )
        self.assertEqual(summary.iloc[0]["average_coverage"], 60.0)
        self.assertEqual(summary.iloc[0]["average_sample_size"], 90)

    def test_gap_chart_handles_multiple_dimension_types(self) -> None:
        chart = dashboard.build_gap_chart(self.df)
        chart_dict = chart.to_dict()
        self.assertEqual(chart_dict["mark"]["type"], "bar")


class DashboardLoadDataTest(unittest.TestCase):
    def test_load_data_coerces_numeric_columns_and_extracts_ci_bounds(self) -> None:
        source_df = pd.DataFrame(
            [
                {
                    "Vaccine": "Tdap",
                    "Geography Type": "States",
                    "Geography": "Ohio",
                    "Survey Year/Influenza Season": "2022",
                    "Dimension Type": "Race and Ethnicity",
                    "Dimension": "White, Non-Hispanic",
                    "Estimate (%)": "70.0",
                    "95% CI (%)": "66.0 to 74.0",
                    "Sample Size": "100",
                },
                {
                    "Vaccine": "Tdap",
                    "Geography Type": "States",
                    "Geography": "Ohio",
                    "Survey Year/Influenza Season": "2022",
                    "Dimension Type": "Race and Ethnicity",
                    "Dimension": "Black, Non-Hispanic",
                    "Estimate (%)": "",
                    "95% CI (%)": "",
                    "Sample Size": "80",
                },
            ]
        )

        with patch("dashboard.pd.read_csv", return_value=source_df):
            loaded = dashboard.load_data()

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded.iloc[0]["CI Lower (%)"], 66.0)
        self.assertEqual(loaded.iloc[0]["CI Upper (%)"], 74.0)
        self.assertEqual(loaded.iloc[0]["Season Sort"], 2022)
        self.assertTrue(math.isclose(loaded.iloc[0]["Sample Size"], 100.0))


if __name__ == "__main__":
    unittest.main()
