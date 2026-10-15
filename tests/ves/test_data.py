from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ertkit import DataWarning, ErrorModel
from ertkit.ves import LayeredModel, Sounding, Survey, forward, read_csv, write_csv

MODEL = LayeredModel(resistivity=[150.0, 30.0, 800.0], thickness=[2.0, 15.0])
SURVEY = Survey.schlumberger(ab2=np.geomspace(2.5, 150.0, 22), mn2=0.5)


class TestSounding:
    def test_derived_quantities(self) -> None:
        rhoa = forward(MODEL, SURVEY)
        data = Sounding(survey=SURVEY, rhoa=rhoa, error=ErrorModel(relative=0.02))
        assert data.n_data == SURVEY.n_readings
        np.testing.assert_allclose(data.resistance, rhoa / SURVEY.geometric_factor)
        np.testing.assert_allclose(data.log_data, np.log(rhoa))
        np.testing.assert_allclose(data.log_std, 0.02)

    def test_synthetic_is_reproducible(self) -> None:
        a = Sounding.synthetic(MODEL, SURVEY, rng=7)
        b = Sounding.synthetic(MODEL, SURVEY, rng=np.random.default_rng(7))
        np.testing.assert_array_equal(a.rhoa, b.rhoa)

    def test_synthetic_noise_has_the_stated_distribution(self) -> None:
        # 2000 soundings of 22 readings: the standardized log residuals should
        # have mean 0 and standard deviation 1. With 44,000 draws the standard
        # errors are 0.005 (mean) and 0.0034 (sd); the bounds are 5 of them.
        # The absolute part keeps every reading far above 3a, so none is removed.
        noise = ErrorModel(relative=0.05, absolute=1e-4)
        clean = forward(MODEL, SURVEY)
        s = noise.log_std(clean / SURVEY.geometric_factor)
        rng = np.random.default_rng(0)
        z = np.concatenate(
            [
                (
                    np.log(Sounding.synthetic(MODEL, SURVEY, noise, rng=rng).rhoa)
                    - np.log(clean)
                )
                / s
                for _ in range(2000)
            ]
        )
        assert abs(z.mean()) < 0.025
        assert abs(z.std() - 1.0) < 0.017

    def test_unusable_readings_are_removed_with_a_warning(self) -> None:
        rhoa = forward(MODEL, SURVEY).copy()
        rhoa[[3, 10]] = [-5.0, 0.0]
        with pytest.warns(
            DataWarning, match=r"indices \[3, 10\] \(AB/2 = 4.487, 17.57 m\)"
        ):
            data = Sounding(survey=SURVEY, rhoa=rhoa)
        assert data.n_data == SURVEY.n_readings - 2
        assert not np.isin(SURVEY.ab2[[3, 10]], data.survey.ab2).any()

    def test_error_model_defaults_and_types(self) -> None:
        rhoa = forward(MODEL, SURVEY)
        assert Sounding(survey=SURVEY, rhoa=rhoa, error=None).error == ErrorModel()  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="ErrorModel"):
            Sounding(survey=SURVEY, rhoa=rhoa, error=0.03)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="Survey"):
            Sounding(survey=SURVEY.ab2, rhoa=rhoa)  # type: ignore[arg-type]

    def test_read_only(self) -> None:
        data = Sounding(survey=SURVEY, rhoa=forward(MODEL, SURVEY))
        with pytest.raises(ValueError, match="read-only"):
            data.rhoa[0] = 1.0

    @pytest.mark.parametrize(
        ("rhoa", "match"),
        [
            (np.ones(3), "need 22"),
            (np.full(22, np.nan), "finite"),
            (np.full(22, -1.0), "no reading is usable"),
        ],
    )
    def test_validation(self, rhoa: np.ndarray, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            Sounding(survey=SURVEY, rhoa=rhoa)


class TestCsv:
    def test_round_trip(self, tmp_path: Path) -> None:
        data = Sounding.synthetic(MODEL, SURVEY, rng=1)
        path = tmp_path / "sounding.csv"
        write_csv(data, path)
        back = read_csv(path, error=data.error)
        np.testing.assert_array_equal(back.rhoa, data.rhoa)
        np.testing.assert_array_equal(back.survey.ab2, data.survey.ab2)
        np.testing.assert_array_equal(back.survey.mn2, data.survey.mn2)

    def test_resistance_column_comments_and_case(self, tmp_path: Path) -> None:
        path = tmp_path / "field.csv"
        path.write_text(
            "# Site 12\n\nAB2,MN2,Resistance,Remarks\n3.0,0.5,10.0,ok\n10.0,1.0,0.5,\n",
            encoding="utf-8",
        )
        data = read_csv(path)
        k = Survey(ab2=[3.0, 10.0], mn2=[0.5, 1.0]).geometric_factor
        np.testing.assert_allclose(data.rhoa, [10.0, 0.5] * k)

    def test_byte_order_mark_and_semicolons(self, tmp_path: Path) -> None:
        # Spreadsheet programs write a byte-order mark; European ones use semicolons.
        path = tmp_path / "excel.csv"
        path.write_bytes("\ufeff# exported\nab2;mn2;rhoa\n3.0;0.5;100.0\n".encode())
        assert read_csv(path).rhoa[0] == 100.0

    def test_empty_spreadsheet_rows_are_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "excel.csv"
        path.write_text(
            "ab2,mn2,rhoa\n3.0,0.5,100.0\n,,\n10.0,0.5,80.0\n,,\n", encoding="utf-8"
        )
        np.testing.assert_array_equal(read_csv(path).rhoa, [100.0, 80.0])

    def test_other_encodings(self, tmp_path: Path) -> None:
        path = tmp_path / "latin1.csv"
        path.write_bytes(
            "# Sondage, Minna \xe9t\xe9\nab2,mn2,rhoa\n3.0,0.5,100.0\n".encode(
                "latin-1"
            )
        )
        with pytest.raises(ValueError, match="encoding='latin-1'"):
            read_csv(path)
        assert read_csv(path, encoding="latin-1").rhoa[0] == 100.0

    @pytest.mark.parametrize(
        ("text", "match"),
        [
            ("", "no header"),
            ("ab2,rhoa\n3,100\n", "missing column 'mn2'"),
            ("ab2,mn2\n3,0.5\n", "'rhoa' or a 'resistance'"),
            ("ab2,mn2,rhoa\n3,0.5,abc\n", "line 2: could not convert"),
            ("ab2,mn2,rhoa\n3,0.5\n", "line 2: too few columns"),
            ("ab2;mn2;rhoa\n3,0;0,5;100\n", "decimal point"),
            ("ab2,AB2,mn2,rhoa\n3,3,0.5,100\n", "duplicate columns"),
            ("ab2,mn2,rhoa\n3,0.5,nan\n", "line 2: values must be finite"),
            ("ab2,mn2,rhoa\n", "no readings"),
            ("ab2,mn2,rhoa\n3,0.5,100\n# c\n2,2,100\n", "line 4: need 0 < MN/2 < AB/2"),
        ],
    )
    def test_errors(self, tmp_path: Path, text: str, match: str) -> None:
        path = tmp_path / "bad.csv"
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match=match):
            read_csv(path)
