"""Deterministic EViews-style text and table rendering."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EViewsReport:
    result: object

    def header(self) -> list[str]:
        r = self.result
        lines = []
        if getattr(r, "title", ""):
            lines.append(str(r.title))
        lines.append(f"Dependent Variable: {getattr(r, 'dependent', 'Y')}")
        if getattr(r, "method", ""):
            lines.append(f"Method: {r.method}")
        if getattr(r, "sample", None):
            lines.append(f"Sample: {r.sample}")
        lines.append(f"Included observations: {getattr(r, 'nobs', 0)}")
        return lines

    def coefficient_table(self) -> pd.DataFrame:
        r = self.result
        return pd.DataFrame({
            "Coefficient": r.params,
            "Std. Error": r.bse,
            "t-Statistic": r.tvalues,
            "Prob.": r.pvalues,
        })

    def text(self, digits: int = 6) -> str:
        r = self.result
        lines = self.header()
        lines.append("")
        lines.append("Variable          Coefficient       Std. Error       t-Statistic       Prob.")
        table = self.coefficient_table()
        for name, row in table.iterrows():
            vals = [
                f"{float(row['Coefficient']): .{digits}f}",
                f"{float(row['Std. Error']): .{digits}f}",
                f"{float(row['t-Statistic']): .{digits}f}",
                f"{float(row['Prob.']): .4f}",
            ]
            lines.append(f"{str(name):<16s}{vals[0]:>17s}{vals[1]:>17s}{vals[2]:>17s}{vals[3]:>13s}")
        lines.append("")
        stats = r.eviews_statistics() if hasattr(r, "eviews_statistics") else r.statistics()
        pairs = [
            ("R-squared", "R-squared", "Akaike info criterion", "Akaike info criterion"),
            ("Adjusted R-squared", "Adjusted R-squared", "Schwarz criterion", "Schwarz criterion"),
            ("S.E. of regression", "S.E. of regression", "Hannan-Quinn criter.", "Hannan-Quinn criterion"),
            ("Sum squared resid", "Sum squared resid", "Durbin-Watson stat", "Durbin-Watson"),
            ("Log likelihood", "Log likelihood", "F-statistic", "F-statistic"),
        ]
        for _, lkey, _, rkey in pairs:
            lv = stats.get(lkey, np.nan)
            rv = stats.get(rkey, np.nan)
            ls = "NA" if not np.isfinite(lv) else f"{lv:.{digits}f}"
            rs = "NA" if not np.isfinite(rv) else f"{rv:.{digits}f}"
            lines.append(f"{lkey:<24s}{ls:>14s}    {rkey:<28s}{rs:>14s}")
        return "\n".join(lines)

    def forecast_table(self, frame: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in ["Forecast", "Std. Error", "Lower", "Upper", "Actual", "Error"] if c in frame.columns]
        return frame.loc[:, cols].copy()

    def diagnostic_text(self, name: str, table) -> str:
        if hasattr(table, "table"):
            data = table.table()
        elif isinstance(table, pd.DataFrame):
            data = table
        elif isinstance(table, dict):
            data = pd.DataFrame([table])
        else:
            data = pd.DataFrame(table)
        return f"{name}\n" + data.to_string(index=False)


def render_eviews(result: object, *, digits: int = 6) -> str:
    return EViewsReport(result).text(digits=digits)
