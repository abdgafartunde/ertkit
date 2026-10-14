"""Sphinx directive that renders the verification report as a table.

The report is written by ``pytest -m verification --verification-report=PATH``
(see ``tests/conftest.py``). The documentation workflow writes it to
``docs/_generated/verification.json`` before building, so every number on the
page comes from tests run for that build.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from docutils import nodes
from docutils.parsers.rst import Directive
from sphinx.application import Sphinx

_COLUMNS = (
    ("Check", 30),
    ("Measured", 10),
    ("Tolerance", 10),
    ("Result", 7),
    ("Why this tolerance", 43),
)


def _row(cells: list[str], header: bool = False) -> nodes.row:
    row = nodes.row()
    for text in cells:
        entry = nodes.entry()
        entry += nodes.paragraph(text=text)
        row += entry
    return row


class VerificationReport(Directive):
    """Render the checks recorded in the verification report."""

    has_content = False

    def run(self) -> list[nodes.Node]:
        env = self.state.document.settings.env
        path = Path(env.srcdir) / env.config.verification_report_path
        env.note_dependency(str(path))
        if not path.is_file():
            return [
                nodes.paragraph(
                    text=(
                        "The verification report is generated for each documentation "
                        "build. To generate it locally, run: pytest -m verification "
                        "--verification-report=docs/_generated/verification.json"
                    )
                )
            ]
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        checks = data["checks"]
        passed = sum(1 for c in checks if c["passed"])
        summary = nodes.paragraph(
            text=(
                f"{passed} of {len(checks)} checks pass, for ertkit {data['ertkit']} "
                f"with Python {data['python']} and numpy {data['numpy']}."
            )
        )
        table = nodes.table()
        group = nodes.tgroup(cols=len(_COLUMNS))
        table += group
        for _, width in _COLUMNS:
            group += nodes.colspec(colwidth=width)
        head = nodes.thead()
        head += _row([name for name, _ in _COLUMNS])
        group += head
        body = nodes.tbody()
        for c in checks:
            body += _row(
                [
                    c["name"],
                    f"{c['measured']:.3g}",
                    f"{c['tolerance']:.3g}",
                    "pass" if c["passed"] else "FAIL",
                    c["justification"],
                ]
            )
        group += body
        return [summary, table]


def setup(app: Sphinx) -> dict[str, Any]:
    app.add_config_value(
        "verification_report_path", "_generated/verification.json", "env"
    )
    app.add_directive("verification-report", VerificationReport)
    return {"version": "1", "parallel_read_safe": True, "parallel_write_safe": True}
