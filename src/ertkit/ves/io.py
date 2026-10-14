"""Reading and writing VES data as CSV files."""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np

from ertkit.noise import ErrorModel
from ertkit.ves.data import Sounding
from ertkit.ves.survey import Survey

__all__ = ["read_csv", "write_csv"]


def read_csv(
    path: str | os.PathLike[str],
    error: ErrorModel | None = None,
    *,
    encoding: str = "utf-8",
) -> Sounding:
    """Read a sounding from a CSV file.

    The file has a header row and one row per reading. Column names are not
    case sensitive. Required columns are ``ab2`` and ``mn2`` (half-spacings in
    meters) and either ``rhoa`` (apparent resistivity in ohm meters) or
    ``resistance`` (transfer resistance in ohms). Lines starting with ``#``
    are comments. Other columns are ignored.

    Parameters
    ----------
    path : str or path-like
        The CSV file.
    error : ErrorModel, optional
        Error model for the data. Default ``ErrorModel()``.
    encoding : str, optional
        Text encoding of the file. Default ``"utf-8"``; files exported by
        some instruments need, for example, ``"latin-1"``.

    Returns
    -------
    Sounding

    Raises
    ------
    ValueError
        If a required column is missing or a value is not a number.

    Examples
    --------
    A file with these lines::

        # Site 12, Minna
        ab2,mn2,rhoa
        1.5,0.5,120.3
        2.5,0.5,98.7

    is read with ``read_csv("site12.csv")``.
    """
    with Path(path).open(encoding=encoding, newline="") as handle:
        lines = [
            line
            for line in handle
            if line.strip() and not line.lstrip().startswith("#")
        ]
    reader = csv.DictReader(lines)
    if reader.fieldnames is None:
        raise ValueError(f"{path}: no header row")
    names = {name.strip().lower(): name for name in reader.fieldnames}
    for required in ("ab2", "mn2"):
        if required not in names:
            raise ValueError(f"{path}: missing column {required!r}")
    if "rhoa" in names:
        value_col, is_resistance = names["rhoa"], False
    elif "resistance" in names:
        value_col, is_resistance = names["resistance"], True
    else:
        raise ValueError(f"{path}: need a 'rhoa' or a 'resistance' column")
    rows = list(reader)
    try:
        ab2 = np.array([float(row[names["ab2"]]) for row in rows])
        mn2 = np.array([float(row[names["mn2"]]) for row in rows])
        values = np.array([float(row[value_col]) for row in rows])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: every value must be a number ({exc})") from exc
    survey = Survey(ab2=ab2, mn2=mn2)
    rhoa = values * survey.geometric_factor if is_resistance else values
    return Sounding(
        survey=survey, rhoa=rhoa, error=ErrorModel() if error is None else error
    )


def write_csv(
    sounding: Sounding, path: str | os.PathLike[str], *, encoding: str = "utf-8"
) -> None:
    """Write a sounding to a CSV file readable by :func:`read_csv`.

    Parameters
    ----------
    sounding : Sounding
        The data to write.
    path : str or path-like
        Destination file; overwritten if it exists.
    encoding : str, optional
        Text encoding. Default ``"utf-8"``.
    """
    with Path(path).open("w", encoding=encoding, newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ab2", "mn2", "rhoa"])
        for ab2, mn2, rhoa in zip(
            sounding.survey.ab2, sounding.survey.mn2, sounding.rhoa, strict=True
        ):
            writer.writerow([repr(float(ab2)), repr(float(mn2)), repr(float(rhoa))])
