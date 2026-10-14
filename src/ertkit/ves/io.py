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
    encoding: str = "utf-8-sig",
) -> Sounding:
    """Read a sounding from a CSV file.

    The file has a header row and one row per reading. Column names are not
    case sensitive. Required columns are ``ab2`` and ``mn2`` (half-spacings in
    meters) and either ``rhoa`` (apparent resistivity in ohm meters) or
    ``resistance`` (transfer resistance in ohms). Lines starting with ``#``
    are comments, and other columns are ignored. Columns may be separated by
    commas, semicolons or tabs; numbers must use a decimal point.

    Parameters
    ----------
    path : str or path-like
        The CSV file.
    error : ErrorModel, optional
        Error model for the data. Default ``ErrorModel()``.
    encoding : str, optional
        Text encoding of the file. The default, ``"utf-8-sig"``, reads UTF-8
        files with or without the byte-order mark that spreadsheet programs
        write; files exported by some instruments need, for example,
        ``"latin-1"``.

    Returns
    -------
    Sounding

    Raises
    ------
    ValueError
        If the header is missing or has duplicate or missing columns, if a
        value is not a number (the message gives the line), or if the
        spacings are invalid (``MN/2`` must be smaller than ``AB/2``) or no
        reading is usable.

    Warns
    -----
    DataWarning
        If readings are removed (see :class:`Sounding`).

    Examples
    --------
    A file with these lines::

        # Site 12, Minna
        ab2,mn2,rhoa
        1.5,0.3,120.3
        2.5,0.3,98.7

    is read with ``read_csv("site12.csv")``.
    """
    with Path(path).open(encoding=encoding, newline="") as handle:
        numbered = [
            (number, line.lstrip("\ufeff"))
            for number, line in enumerate(handle, start=1)
            if line.strip() and not line.lstrip("\ufeff").lstrip().startswith("#")
        ]
    if not numbered:
        raise ValueError(f"{path}: no header row")
    try:
        delimiter = csv.Sniffer().sniff(numbered[0][1], delimiters=",;\t").delimiter
    except csv.Error:
        delimiter = ","
    rows = list(csv.reader([line for _, line in numbered], delimiter=delimiter))
    header = [name.strip().lower() for name in rows[0]]
    duplicated = sorted({name for name in header if name and header.count(name) > 1})
    if duplicated:
        raise ValueError(f"{path}: duplicate columns {duplicated}")
    names = {name: k for k, name in enumerate(header)}
    for required in ("ab2", "mn2"):
        if required not in names:
            raise ValueError(
                f"{path}: missing column {required!r} (columns found: {header})"
            )
    if "rhoa" in names:
        value_col, is_resistance = names["rhoa"], False
    elif "resistance" in names:
        value_col, is_resistance = names["resistance"], True
    else:
        raise ValueError(f"{path}: need a 'rhoa' or a 'resistance' column")
    columns = (names["ab2"], names["mn2"], value_col)
    ab2, mn2, values = [], [], []
    for (number, _), row in zip(numbered[1:], rows[1:], strict=True):
        if len(row) <= max(columns):
            raise ValueError(f"{path}, line {number}: too few columns")
        try:
            a, m, v = (float(row[c]) for c in columns)
        except ValueError as exc:
            hint = (
                " (numbers must use a decimal point, not a comma)"
                if any("," in cell for cell in row)
                else ""
            )
            raise ValueError(f"{path}, line {number}: {exc}{hint}") from exc
        if not all(np.isfinite([a, m, v])):
            raise ValueError(f"{path}, line {number}: values must be finite")
        ab2.append(a)
        mn2.append(m)
        values.append(v)
    survey = Survey(ab2=np.array(ab2), mn2=np.array(mn2))
    rhoa = (
        np.array(values) * survey.geometric_factor
        if is_resistance
        else np.array(values)
    )
    return Sounding(
        survey=survey, rhoa=rhoa, error=ErrorModel() if error is None else error
    )


def write_csv(
    sounding: Sounding, path: str | os.PathLike[str], *, encoding: str = "utf-8"
) -> None:
    """Write a sounding to a CSV file readable by :func:`read_csv`.

    The error model is not written; pass it to :func:`read_csv` again.

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
