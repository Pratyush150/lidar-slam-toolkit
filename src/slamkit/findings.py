"""Shared result types for every diagnostic in :mod:`slamkit`.

A diagnostic never returns a bare boolean.  A boolean tells you *that*
something is wrong; it does not tell you which knob to turn.  Every check in
this toolkit returns :class:`Finding` objects that carry four things:

``code``
    A stable machine-readable identifier so ``slam-doctor --json`` output can
    be diffed between runs.
``message``
    What was measured, with the number that triggered it.
``symptom``
    What this defect looks like *in RViz*, because that is how the problem
    was reported to you in the first place.
``fix``
    The concrete edit to make.

Severity ordering is used to rank the final report: the top line of a
diagnosis should be the thing that, once fixed, makes the other complaints
disappear.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional

__all__ = ["Severity", "Finding", "Report"]


class Severity(enum.IntEnum):
    """Ranked severity. Higher integer sorts first in a report."""

    OK = 0
    """Check ran and passed. Kept in the report so you can see what was tested."""

    INFO = 1
    """Worth knowing, not a defect."""

    WARN = 2
    """Will degrade accuracy. SLAM still runs."""

    ERROR = 3
    """Will produce a visibly wrong map. Fix before tuning anything."""

    CRITICAL = 4
    """SLAM cannot work at all until this is fixed."""

    def label(self) -> str:
        return self.name


@dataclass
class Finding:
    """One diagnostic result.

    Parameters
    ----------
    code:
        Stable identifier, e.g. ``"EXTRINSIC_TRANSPOSED"``.
    severity:
        See :class:`Severity`.
    message:
        Measured fact, including the number.
    symptom:
        The visible behaviour this defect causes.
    fix:
        The action to take.
    data:
        Optional machine-readable payload (measured values, indices).
    """

    code: str
    severity: Severity
    message: str
    symptom: str = ""
    fix: str = ""
    data: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True when this finding does not indicate a defect."""
        return self.severity <= Severity.INFO

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["severity"] = self.severity.name
        return d

    def __str__(self) -> str:  # pragma: no cover - formatting only
        head = f"[{self.severity.name:<8}] {self.code}: {self.message}"
        parts = [head]
        if self.symptom:
            parts.append(f"           symptom: {self.symptom}")
        if self.fix:
            parts.append(f"           fix:     {self.fix}")
        return "\n".join(parts)


@dataclass
class Report:
    """An ordered collection of :class:`Finding` objects."""

    title: str = ""
    findings: List[Finding] = field(default_factory=list)

    def add(self, finding: Optional[Finding]) -> "Report":
        """Append ``finding`` unless it is ``None``. Returns ``self``."""
        if finding is not None:
            self.findings.append(finding)
        return self

    def extend(self, findings: Iterable[Finding]) -> "Report":
        for f in findings:
            self.add(f)
        return self

    @property
    def problems(self) -> List[Finding]:
        """Findings at WARN or above, worst first."""
        bad = [f for f in self.findings if not f.ok]
        return sorted(bad, key=lambda f: -int(f.severity))

    @property
    def worst(self) -> Severity:
        if not self.findings:
            return Severity.OK
        return max(f.severity for f in self.findings)

    def ranked(self) -> List[Finding]:
        """All findings, worst first, stable within a severity level."""
        return sorted(self.findings, key=lambda f: -int(f.severity))

    def has(self, code: str) -> bool:
        """True if any finding carries ``code``."""
        return any(f.code == code for f in self.findings)

    def get(self, code: str) -> Optional[Finding]:
        """First finding with ``code``, else ``None``."""
        for f in self.findings:
            if f.code == code:
                return f
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "worst_severity": self.worst.name,
            "findings": [f.to_dict() for f in self.ranked()],
        }

    def __len__(self) -> int:
        return len(self.findings)

    def __iter__(self):
        return iter(self.findings)

    def __str__(self) -> str:  # pragma: no cover - formatting only
        lines = []
        if self.title:
            lines.append(self.title)
            lines.append("-" * len(self.title))
        for f in self.ranked():
            lines.append(str(f))
        return "\n".join(lines)
