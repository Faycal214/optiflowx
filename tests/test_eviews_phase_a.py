"""Numerical parity checks against the official EViews Time Series tutorial.

The raw Data.xlsx file is deliberately not vendored into StochX. Set
STOCHX_EVIEWS_DATA to its local path to run these checks. Reference values are
stored as display-preserving strings so rounded EViews output is not mistaken
for machine-precision ground truth.
"""

from __future__ import annotations

import json
import os
from decimal import Decimal, InvalidOperation
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET

import numpy as np
import pytest

from stochx.timeseries import Workfile


ROOT = Path(__file__).resolve().parent
EXPECTED = ROOT / "fixtures" / "eviews_phase_a_expected.json"
DATA = os.environ.get("STOCHX_EVIEWS_DATA")


def _load_eviews_xlsx(path: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    """Load the four tutorial columns without adding an XLSX runtime dependency."""
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with ZipFile(path) as archive:
        strings_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        shared = [
            "".join(t.text or "" for t in item.findall(".//a:t", ns))
            for item in strings_root.findall("a:si", ns)
        ]
        sheet_root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))

    rows: dict[int, dict[str, object]] = {}
    for row in sheet_root.findall(".//a:sheetData/a:row", ns):
        row_number = int(row.attrib["r"])
        values: dict[str, object] = {}
        for cell in row.findall("a:c", ns):
            value = cell.find("a:v", ns)
            if value is None:
                continue
            raw = value.text or ""
            column = cell.attrib["r"][0]
            if cell.attrib.get("t") == "s":
                values[column] = shared[int(raw)]
            else:
                values[column] = float(raw)
        rows[row_number] = values

    dates: list[str] = []
    columns = {"M1": [], "Tbill": [], "CPI": [], "IP": []}
    for row_number in sorted(rows):
        values = rows[row_number]
        date = values.get("A")
        if not isinstance(date, str) or not date[:4].isdigit():
            continue
        if not all(key in values for key in ("B", "C", "D", "E")):
            continue
        dates.append(date)
        columns["M1"].append(float(values["B"]))
        columns["Tbill"].append(float(values["C"]))
        columns["CPI"].append(float(values["D"]))
        columns["IP"].append(float(values["E"]))

    return dates, {key: np.asarray(value, dtype=float) for key, value in columns.items()}


def _display_tolerance(expected_text: str) -> float:
    """Return half of the last displayed decimal unit in an EViews string."""
    try:
        value = Decimal(expected_text)
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"invalid numeric reference {expected_text!r}") from exc
    exponent = value.as_tuple().exponent
    if exponent >= 0:
        return 0.5
    return float(Decimal("0.5") * (Decimal(10) ** exponent))


def _assert_eviews_value(
    actual: float,
    expected_text: str,
    *,
    kind: str,
    rel: float = 1e-10,
    atol_override: float | None = None,
) -> None:
    """Compare a numerical result against an EViews display value by field type."""
    expected = float(expected_text)
    if atol_override is not None:
        atol = atol_override
    elif kind == "coefficient" or kind == "std_error":
        atol = _display_tolerance(expected_text)
    elif kind == "t_stat":
        atol = _display_tolerance(expected_text)
    elif kind == "p_value":
        # EViews often prints highly significant p-values as 0.0/0.0000;
        # preserve a tight, practical probability tolerance in that case.
        atol = max(_display_tolerance(expected_text), 5e-5)
    elif kind == "statistic":
        atol = _display_tolerance(expected_text)
    else:
        raise ValueError(f"unknown parity field kind: {kind!r}")

    assert actual == pytest.approx(expected, abs=atol, rel=rel)


def _build_workfile(path: Path) -> Workfile:
    dates, values = _load_eviews_xlsx(path)
    wf = Workfile(frequency="M")
    for name, column in values.items():
        wf.add(name, column, index=dates)
    return wf


@pytest.mark.parity
@pytest.mark.skipif(not DATA or not Path(DATA).exists(), reason="Set STOCHX_EVIEWS_DATA to the EViews tutorial Data.xlsx file")
def test_eviews_phase_a_eq01_eq02_eq02a_eq03_numerical_parity():
    with EXPECTED.open("r", encoding="utf-8") as handle:
        expected = json.load(handle)["equations"]

    wf = _build_workfile(Path(DATA))
    specs = {
        "EQ01": expected["EQ01"]["specification"],
        "EQ02": expected["EQ02"]["specification"],
        "EQ02A": expected["EQ02A"]["specification"],
        "EQ03": expected["EQ03"]["specification"],
    }

    for name, specification in specs.items():
        result = wf.ls(specification, name=name)
        reference = expected[name]
        assert result.nobs == reference["nobs"]

        result_table = result.table()
        for term, fields in reference["coefficients"].items():
            assert term in result_table.index, f"missing EViews regressor {term} in {name}"
            row = result_table.loc[term]
            _assert_eviews_value(float(row["Coefficient"]), fields["coefficient"], kind="coefficient")
            _assert_eviews_value(float(row["Std. Error"]), fields["std_error"], kind="std_error")
            _assert_eviews_value(float(row["t-Statistic"]), fields["t_stat"], kind="t_stat")
            _assert_eviews_value(float(row["Prob."]), fields["p_value"], kind="p_value")

        stats = result.statistics()
        for label, expected_value in reference["statistics"].items():
            _assert_eviews_value(stats[label], expected_value, kind="statistic")


def test_eviews_phase_a_range_expansion_and_observation_count_without_data():
    x = np.arange(1.0, 30.0)
    wf = Workfile(frequency="M")
    wf.add("M1", 100.0 + x, index=list(range(len(x))))
    wf.add("CPI", 20.0 + 0.2 * x, index=list(range(len(x))))

    result = wf.ls("M1 C CPI(0 to -12)", name="EQ02A")
    assert result.nobs == 17
    assert list(result.table().index) == ["C"] + ["CPI" if lag == 0 else f"CPI(-{lag})" for lag in range(13)]
