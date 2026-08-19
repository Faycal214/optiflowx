import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from stochx.timeseries import arma, correlogram, plot_eviews_correlogram, white_noise


def _assert_band_lines(axis, lower, upper):
    assert len(axis.lines) == 3
    np.testing.assert_allclose(axis.lines[1].get_ydata(), upper)
    np.testing.assert_allclose(axis.lines[2].get_ydata(), lower)
    np.testing.assert_allclose(axis.lines[1].get_xdata(), np.arange(1, len(upper) + 1))
    np.testing.assert_allclose(axis.lines[2].get_xdata(), np.arange(1, len(lower) + 1))
    assert len(axis.collections) == 1
    np.testing.assert_allclose(axis.collections[0].get_segments()[0][0], [1.0, 0.0])


def test_stage8_9_ordinary_plot_uses_frozen_ac_pac_and_bands():
    result = correlogram(white_noise(120, rng=21), nlags=8, model_df=0)
    before = {
        "ac": result.ac.copy(),
        "pac": result.pac.copy(),
        "ac_lower": result.ac_lower.copy(),
        "ac_upper": result.ac_upper.copy(),
        "pac_lower": result.pac_lower.copy(),
        "pac_upper": result.pac_upper.copy(),
    }

    fig, (ac_axis, pac_axis) = plot_eviews_correlogram(result, show=False)

    assert len(fig.axes) == 2
    assert ac_axis.get_title() == "AC"
    assert pac_axis.get_title() == "PAC"
    _assert_band_lines(ac_axis, result.ac_lower, result.ac_upper)
    _assert_band_lines(pac_axis, result.pac_lower, result.pac_upper)
    np.testing.assert_array_equal(result.ac, before["ac"])
    np.testing.assert_array_equal(result.pac, before["pac"])
    np.testing.assert_array_equal(result.ac_lower, before["ac_lower"])
    np.testing.assert_array_equal(result.ac_upper, before["ac_upper"])
    np.testing.assert_array_equal(result.pac_lower, before["pac_lower"])
    np.testing.assert_array_equal(result.pac_upper, before["pac_upper"])
    plt.close(fig)


def test_stage8_9_residual_plot_preserves_model_df_and_band_contract():
    result = correlogram(arma(p=1, q=1, phi=[0.45], theta=[0.25], n=160, rng=7), nlags=6, model_df=2)
    fig, axes = plot_eviews_correlogram(result, title="Residual Correlogram", show=False)

    assert result.model_df == 2
    assert axes[0].get_title() == "Residual Correlogram — AC"
    assert axes[1].get_title() == "Residual Correlogram — PAC"
    assert fig._suptitle.get_text() == "Residual Correlogram"
    np.testing.assert_allclose(axes[0].lines[1].get_ydata(), result.ac_upper)
    np.testing.assert_allclose(axes[0].lines[2].get_ydata(), result.ac_lower)
    np.testing.assert_allclose(axes[1].lines[1].get_ydata(), result.pac_upper)
    np.testing.assert_allclose(axes[1].lines[2].get_ydata(), result.pac_lower)
    plt.close(fig)


def test_stage8_9_custom_axes_are_supported_without_recomputation():
    result = correlogram(white_noise(80, rng=9), nlags=5)
    fig, axes_array = plt.subplots(2, 1)
    returned_fig, returned_axes = plot_eviews_correlogram(result, axes=axes_array, show=False)

    assert returned_fig is fig
    assert returned_axes == (axes_array[0], axes_array[1])
    np.testing.assert_allclose(axes_array[0].lines[1].get_ydata(), result.ac_upper)
    np.testing.assert_allclose(axes_array[1].lines[2].get_ydata(), result.pac_lower)
    plt.close(fig)


def test_stage8_9_plot_validates_result_and_axes():
    with pytest.raises(TypeError, match="CorrelogramResult"):
        plot_eviews_correlogram(object(), show=False)

    result = correlogram(white_noise(60, rng=1), nlags=4)
    with pytest.raises(ValueError, match="exactly two"):
        plot_eviews_correlogram(result, axes=[plt.subplots()[1]], show=False)
    plt.close("all")
