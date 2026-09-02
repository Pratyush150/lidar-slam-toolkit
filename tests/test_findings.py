"""The Finding / Report result types that every diagnostic returns."""

import json

from slamkit.findings import Finding, Report, Severity


def test_severity_is_ordered():
    assert Severity.CRITICAL > Severity.ERROR > Severity.WARN > Severity.INFO > Severity.OK
    assert max([Severity.WARN, Severity.CRITICAL, Severity.OK]) == Severity.CRITICAL


def test_finding_ok_property_splits_at_warn():
    assert Finding("A", Severity.OK, "m").ok
    assert Finding("A", Severity.INFO, "m").ok
    assert not Finding("A", Severity.WARN, "m").ok
    assert not Finding("A", Severity.CRITICAL, "m").ok


def test_report_ranks_worst_first():
    rep = Report(title="t")
    rep.add(Finding("LOW", Severity.WARN, "w"))
    rep.add(Finding("HIGH", Severity.CRITICAL, "c"))
    rep.add(Finding("MID", Severity.ERROR, "e"))
    assert [f.code for f in rep.ranked()] == ["HIGH", "MID", "LOW"]
    assert rep.worst == Severity.CRITICAL


def test_report_problems_excludes_ok_and_info():
    rep = Report()
    rep.extend([Finding("A", Severity.OK, "a"), Finding("B", Severity.INFO, "b"),
                Finding("C", Severity.WARN, "c")])
    assert [f.code for f in rep.problems] == ["C"]
    assert len(rep) == 3


def test_report_add_ignores_none():
    rep = Report()
    rep.add(None).add(Finding("A", Severity.OK, "a"))
    assert len(rep) == 1


def test_report_lookup_helpers():
    rep = Report()
    rep.add(Finding("CODE_X", Severity.WARN, "m", data={"k": 1}))
    assert rep.has("CODE_X")
    assert not rep.has("CODE_Y")
    assert rep.get("CODE_X").data["k"] == 1
    assert rep.get("CODE_Y") is None


def test_report_is_iterable_and_serialisable():
    rep = Report(title="demo")
    rep.add(Finding("A", Severity.ERROR, "msg", symptom="s", fix="f"))
    assert [f.code for f in rep] == ["A"]
    d = rep.to_dict()
    assert d["worst_severity"] == "ERROR"
    assert d["findings"][0]["severity"] == "ERROR"
    assert json.loads(json.dumps(d))["title"] == "demo"


def test_empty_report_is_ok():
    rep = Report()
    assert rep.worst == Severity.OK
    assert rep.problems == []


def test_finding_str_includes_symptom_and_fix():
    text = str(Finding("A", Severity.WARN, "measured 3.2", symptom="map tilts",
                       fix="check the extrinsic"))
    assert "measured 3.2" in text
    assert "map tilts" in text
    assert "check the extrinsic" in text
