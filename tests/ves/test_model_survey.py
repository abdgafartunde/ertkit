from __future__ import annotations

import numpy as np
import pytest

from ertkit.ves import LayeredModel, Survey


class TestLayeredModel:
    def test_properties(self) -> None:
        m = LayeredModel(resistivity=[150.0, 30.0, 800.0], thickness=[2.0, 15.0])
        assert m.n_layers == 3
        np.testing.assert_allclose(m.depths, [2.0, 17.0])
        np.testing.assert_allclose(m.conductance, [2.0 / 150.0, 15.0 / 30.0])
        np.testing.assert_allclose(m.transverse_resistance, [300.0, 450.0])

    def test_half_space(self) -> None:
        m = LayeredModel(resistivity=[100.0], thickness=[])
        assert m.n_layers == 1
        assert m.depths.size == 0

    def test_arrays_are_copied_and_read_only(self) -> None:
        rho = np.array([10.0, 20.0])
        m = LayeredModel(resistivity=rho, thickness=[1.0])
        rho[0] = 99.0
        assert m.resistivity[0] == 10.0
        with pytest.raises(ValueError, match="read-only"):
            m.resistivity[0] = 5.0

    def test_frozen(self) -> None:
        m = LayeredModel(resistivity=[10.0], thickness=[])
        with pytest.raises(AttributeError):
            m.resistivity = np.array([1.0])  # type: ignore[misc]

    @pytest.mark.parametrize(
        ("rho", "h", "match"),
        [
            ([10.0, -1.0], [1.0], "resistivity"),
            ([10.0, np.inf], [1.0], "resistivity"),
            ([10.0, 20.0], [0.0], "thickness"),
            ([10.0, 20.0], [1.0, 2.0], "need 1 thicknesses"),
            ([[10.0, 20.0]], [1.0], "one-dimensional"),
        ],
    )
    def test_validation(self, rho: list[float], h: list[float], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            LayeredModel(resistivity=rho, thickness=h)


class TestSurvey:
    def test_schlumberger_scalar_mn2(self) -> None:
        s = Survey.schlumberger(ab2=[3.0, 10.0, 30.0], mn2=0.5)
        assert s.n_readings == 3
        np.testing.assert_allclose(s.mn2, 0.5)
        np.testing.assert_allclose(s.geometric_factor, np.pi * (s.ab2**2 - 0.25) / 1.0)

    def test_wenner(self) -> None:
        a = np.array([1.0, 3.0])
        s = Survey.wenner(a)
        np.testing.assert_allclose(s.geometric_factor, 2 * np.pi * a)

    def test_read_only(self) -> None:
        s = Survey.schlumberger(ab2=[3.0], mn2=1.0)
        with pytest.raises(ValueError, match="read-only"):
            s.ab2[0] = 1.0

    @pytest.mark.parametrize(
        ("ab2", "mn2", "match"),
        [
            ([3.0], [3.0], "smaller"),
            ([3.0], [-1.0], "mn2"),
            ([np.nan], [1.0], "ab2"),
            ([], 0.5, "non-empty"),
        ],
    )
    def test_validation(self, ab2: list[float], mn2: object, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            Survey.schlumberger(ab2=ab2, mn2=mn2)  # type: ignore[arg-type]
