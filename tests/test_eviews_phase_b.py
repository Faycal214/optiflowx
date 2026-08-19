"""Structural Phase B checks against the official EViews tutorial.

The raw Data.xlsx is supplied externally through STOCHX_EVIEWS_DATA. Numerical
ARMA parity is deliberately not asserted until the optimizer/likelihood
conventions are matched to EViews; these tests lock the EViews specification,
AR/MA parsing, sample handling, parameter naming, and diagnostics API first.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET

import numpy as np
import pytest

from stochx.timeseries import Workfile, breusch_godfrey_raw, parse_error_terms

ROOT = Path(__file__).resolve().parent
EXPECTED = ROOT / "fixtures" / "eviews_phase_b_expected.json"
DATA = os.environ.get("STOCHX_EVIEWS_DATA")


def _load_data(path: Path) -> Workfile:
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with ZipFile(path) as archive:
        strings_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        shared = ["".join(t.text or "" for t in item.findall(".//a:t", ns)) for item in strings_root.findall("a:si", ns)]
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    rows = {}
    for row in sheet.findall(".//a:sheetData/a:row", ns):
        values = {}
        for cell in row.findall("a:c", ns):
            value = cell.find("a:v", ns)
            if value is None:
                continue
            raw = value.text or ""
            col = cell.attrib["r"][0]
            values[col] = shared[int(raw)] if cell.attrib.get("t") == "s" else float(raw)
        rows[int(row.attrib["r"])] = values
    wf = Workfile(frequency="M")
    columns = {"M1": [], "Tbill": [], "CPI": [], "IP": []}
    dates = []
    for row in rows.values():
        date = row.get("A")
        if not isinstance(date, str) or not date[:4].isdigit():
            continue
        if not all(c in row for c in "BCDE"):
            continue
        dates.append(date)
        for key, col in zip(columns, "BCDE"):
            columns[key].append(float(row[col]))
    for key, values in columns.items():
        wf.add(key, values, index=dates)
    return wf


def test_phase_b_error_process_parser():
    regressors, process = parse_error_terms(["C", "LOG(M1)", "AR(1 to 2)", "MA(1 to 3)"])
    assert regressors == ["C", "LOG(M1)"]
    assert process.p == (1, 2)
    assert process.q == (1, 2, 3)
    assert process.order == (2, 3)


@pytest.mark.skipif(not DATA or not Path(DATA).exists(), reason="Set STOCHX_EVIEWS_DATA to Data.xlsx")
def test_phase_b_eviews_equation_specifications_and_orders():
    expected = json.loads(EXPECTED.read_text(encoding="utf-8"))["equations"]
    wf = _load_data(Path(DATA))
    for name, reference in expected.items():
        result = wf.ls(reference["specification"], name=name)
        assert result.nobs == reference["nobs"]
        assert result.error_process.p == tuple(reference["orders"]["ar"])
        assert result.error_process.q == tuple(reference["orders"]["ma"])
        assert result.method == "ARMA Maximum Likelihood (BFGS)"
        for term in reference["reference"]:
            assert term in result.params.index
        assert np.all(np.isfinite(result.params.to_numpy()))
        assert np.all(np.isfinite(result.bse.to_numpy()))


@pytest.mark.skipif(not DATA or not Path(DATA).exists(), reason="Set STOCHX_EVIEWS_DATA to Data.xlsx")
def test_phase_b_eq01_breusch_godfrey_api_on_original_regression():
    wf = _load_data(Path(DATA))
    eq = wf.ls("LOG(M1) C LOG(IP) LOG(CPI) TBILL", name="EQ01")
    out = breusch_godfrey_raw(eq.result, lags=1)
    assert set(out) == {"LM statistic", "LM p-value", "F-statistic", "F p-value"}
    assert all(np.isfinite(value) for value in out.values())
