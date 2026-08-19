"""Numerical parity checks against the official EViews Time Series tutorial."""

from __future__ import annotations

import json
import os
from decimal import Decimal
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


def _display_atol(text: str) -> float:
    """Half a unit of the last displayed decimal place."""
    exponent = Decimal(text).as_tuple().exponent
    if exponent >= 0:
        return 0.5
    return float(Decimal("0.5") * (Decimal(10) ** exponent))


def _assert_display(actual: float, expected_text: str, *, rel: float = 1e-9) -> None:
    expected = float(expected_text)
    assert actual == pytest.approx(expected, abs=_display_atol(expected_text), rel=rel)


def _start_params(reference: dict[str, dict[str, str]]) -> np.ndarray:
    """Build the published EViews coefficient-order start vector for parity isolation."""
    ordered = [float(fields["coefficient"]) for name, fields in reference.items() if name != "SIGMASQ"]
    ordered.append(float(reference["SIGMASQ"]["coefficient"]))
    return np.asarray(ordered, dtype=float)


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
        result = wf.ls(reference["specification"], name=name, start_params=_start_params(reference["coefficients"]))
        assert result.nobs == reference["nobs"]
        assert result.error_process.p == tuple(reference["orders"]["ar"])
        assert result.error_process.q == tuple(reference["orders"]["ma"])
        assert result.method == "ARMA Maximum Likelihood (BFGS)"
        assert np.all(np.isfinite(result.params.to_numpy()))


@pytest.mark.skipif(not DATA or not Path(DATA).exists(), reason="Set STOCHX_EVIEWS_DATA to Data.xlsx")
def test_phase_b_eq18_eq19_eq20_eq21_numerical_parity():
    expected = json.loads(EXPECTED.read_text(encoding="utf-8"))["equations"]
    wf = _load_data(Path(DATA))

    for name, reference in expected.items():
        result = wf.ls(reference["specification"], name=name, start_params=_start_params(reference["coefficients"]))
        table = result.table()

        # Coefficients remain a tight primary parity target.
        for term, fields in reference["coefficients"].items():
            assert term in table.index, f"missing EViews coefficient {term} in {name}"
            _assert_display(float(table.loc[term, "Coefficient"]), fields["coefficient"], rel=2e-8)

        _assert_display(float(result.params["SIGMASQ"]), reference["coefficients"]["SIGMASQ"]["coefficient"], rel=2e-8)

        stats = result.statistics()
        for label in ("Akaike info criterion", "Schwarz criterion", "Hannan-Quinn criterion"):
            _assert_display(stats[label], reference["statistics"][label], rel=2e-8)

        roots = result.roots_report()
        expected_ar = reference["roots"]["ar"]
        expected_ma = reference["roots"]["ma"]
        assert len(roots["Inverted AR Roots"]) == len(expected_ar)
        assert len(roots["Inverted MA Roots"]) == len(expected_ma)

        for actual, text in zip(roots["Inverted AR Roots"], expected_ar):
            _assert_display(float(np.real(actual)), text, rel=2e-2)

        for actual, text in zip(roots["Inverted MA Roots"], expected_ma):
            if "i" not in text:
                _assert_display(float(np.real(actual)), text, rel=2e-2)
                continue
            sign = "+" if "+" in text else "-"
            real_text, imag_text = text.replace("i", "").split(sign)
            _assert_display(float(np.real(actual)), real_text, rel=2e-2)
            _assert_display(abs(float(np.imag(actual))), imag_text, rel=2e-2)


@pytest.mark.skipif(not DATA or not Path(DATA).exists(), reason="Set STOCHX_EVIEWS_DATA to Data.xlsx")
def test_phase_b_eviews_opg_covariance_and_inference():
    expected = json.loads(EXPECTED.read_text(encoding="utf-8"))["equations"]
    wf = _load_data(Path(DATA))

    for name, reference in expected.items():
        result = wf.ls(reference["specification"], name=name, start_params=_start_params(reference["coefficients"]))
        assert result.covariance_method == "outer product of gradients (OPG)"
        covariance = result.covariance_matrix()
        bse = result.bse
        tvalues = result.tvalues
        pvalues = result.pvalues

        # Published EViews screens expose standard errors, but not the full
        # covariance matrix. Assert each diagonal variance against SE^2 and
        # independently assert the resulting SE/t/p columns.
        for term, fields in reference["coefficients"].items():
            assert term in covariance.index, f"missing OPG covariance row {term} in {name}"
            se_text = fields["std_error"]
            expected_se = float(se_text)
            expected_var = expected_se ** 2
            variance_atol = max(2.0 * expected_se * _display_atol(se_text), 1e-10)
            assert float(covariance.loc[term, term]) == pytest.approx(
                expected_var,
                abs=variance_atol,
                rel=2e-6,
            )
            _assert_display(float(bse.loc[term]), se_text, rel=2e-6)
            _assert_display(float(tvalues.loc[term]), fields["t_stat"], rel=2e-6)
            _assert_display(float(pvalues.loc[term]), fields["p_value"], rel=2e-4)

        assert np.allclose(covariance.to_numpy(), covariance.to_numpy().T, atol=1e-12, rtol=0.0)


@pytest.mark.skipif(not DATA or not Path(DATA).exists(), reason="Set STOCHX_EVIEWS_DATA to Data.xlsx")
def test_phase_b_eq01_breusch_godfrey_api_on_original_regression():
    wf = _load_data(Path(DATA))
    eq = wf.ls("LOG(M1) C LOG(IP) LOG(CPI) TBILL", name="EQ01")
    out = breusch_godfrey_raw(eq.result, lags=1)
    assert set(out) == {"LM statistic", "LM p-value", "F-statistic", "F p-value"}
    assert all(np.isfinite(value) for value in out.values())
