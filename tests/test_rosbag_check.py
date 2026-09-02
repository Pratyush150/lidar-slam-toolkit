"""Bag checks: topic list, rates, frame_ids and TF completeness -- all offline."""

import json

import pytest

from slamkit.findings import Severity
from slamkit.rosbag_check import (
    CARTOGRAPHER_2D_EXPECTATIONS,
    LIO_SAM_EXPECTATIONS,
    BagInfo,
    TopicInfo,
    build_tf_graph,
    check_bag,
    check_frame_ids,
    check_tf_completeness,
    check_topics,
    load_any,
    load_json,
    parse_ros2_bag_info_text,
)

BAG_INFO_TEXT = """
Files:             rosbag2_2025_01_01_0.db3
Bag size:          1.2 GiB
Storage id:        sqlite3
Duration:          100.000s
Start:             Jan  1 2025 10:00:00.000 (1735725600.000)
End:               Jan  1 2025 10:01:40.000 (1735725700.000)
Messages:          21000
Topic information: Topic: /points | Type: sensor_msgs/msg/PointCloud2 | Count: 1000 | Serialization Format: cdr
                   Topic: /imu/data | Type: sensor_msgs/msg/Imu | Count: 20000 | Serialization Format: cdr
"""

GOOD_JSON = {
    "path": "good_bag",
    "duration_s": 100.0,
    "message_count": 21000,
    "topics": [
        {"name": "/points", "type": "sensor_msgs/msg/PointCloud2", "count": 1000},
        {"name": "/imu/data", "type": "sensor_msgs/msg/Imu", "count": 20000},
    ],
    "frame_ids": {"/points": "velodyne", "/imu/data": "imu_link"},
    "tf": [
        {"parent": "base_link", "child": "velodyne", "static": True},
        {"parent": "base_link", "child": "imu_link", "static": True},
        {"parent": "odom", "child": "base_link", "static": False},
    ],
}


# ------------------------------------------------------------------ loading
def test_parse_ros2_bag_info_text():
    info = parse_ros2_bag_info_text(BAG_INFO_TEXT)
    assert info.duration_s == 100.0
    assert info.message_count == 21000
    assert len(info.topics) == 2
    assert info.topic("/points").frequency_hz == pytest.approx(10.0)
    assert info.topic("/imu/data").frequency_hz == pytest.approx(200.0)
    assert info.source == "text"


def test_parse_rejects_text_that_is_not_bag_info():
    with pytest.raises(ValueError):
        parse_ros2_bag_info_text("hello world")


def test_load_json_from_dict_string_and_file(tmp_path):
    a = load_json(GOOD_JSON)
    b = load_json(json.dumps(GOOD_JSON))
    p = tmp_path / "bag.json"
    p.write_text(json.dumps(GOOD_JSON))
    c = load_json(str(p))
    for info in (a, b, c):
        assert len(info.topics) == 2
        assert info.topic("/points").frame_id == "velodyne"
        assert len(info.tf_edges) == 3


def test_load_any_dispatches_on_content(tmp_path):
    txt = tmp_path / "info.txt"
    txt.write_text(BAG_INFO_TEXT)
    js = tmp_path / "info.json"
    js.write_text(json.dumps(GOOD_JSON))
    assert load_any(str(txt)).source == "text"
    assert load_any(str(js)).source == "json"
    assert load_any(GOOD_JSON).source == "json"
    with pytest.raises(FileNotFoundError):
        load_any(str(tmp_path / "nope"))


def test_find_matches_name_or_type():
    info = load_json(GOOD_JSON)
    assert len(info.find("PointCloud2")) == 1
    assert len(info.find("^/imu")) == 1


# ------------------------------------------------------------------- topics
def test_topic_check_passes_a_good_bag():
    findings = check_topics(load_json(GOOD_JSON), LIO_SAM_EXPECTATIONS)
    codes = {f.code for f in findings}
    assert "BAG_OK_POINT_CLOUD" in codes
    assert "BAG_OK_IMU" in codes
    assert not any(f.severity >= Severity.ERROR for f in findings)


def test_topic_check_catches_a_missing_imu():
    data = dict(GOOD_JSON)
    data["topics"] = [GOOD_JSON["topics"][0]]
    findings = check_topics(load_json(data), LIO_SAM_EXPECTATIONS)
    missing = [f for f in findings if f.code == "BAG_MISSING_IMU"]
    assert missing and missing[0].severity == Severity.CRITICAL
    assert "never receives a callback" in missing[0].symptom


def test_topic_check_catches_a_slow_imu():
    data = json.loads(json.dumps(GOOD_JSON))
    data["topics"][1]["count"] = 2000        # 20 Hz over 100 s
    findings = check_topics(load_json(data), LIO_SAM_EXPECTATIONS)
    low = [f for f in findings if f.code == "BAG_LOW_RATE_IMU"]
    assert low and low[0].severity == Severity.ERROR


def test_optional_topics_are_only_informational():
    findings = check_topics(load_json(GOOD_JSON), LIO_SAM_EXPECTATIONS)
    gps = [f for f in findings if "GPS" in f.code]
    assert gps and gps[0].severity == Severity.INFO


def test_cartographer_expectations_differ_from_lio_sam():
    """Cartographer 2D wants a LaserScan; a PointCloud2-only bag is not enough."""
    findings = check_topics(load_json(GOOD_JSON), CARTOGRAPHER_2D_EXPECTATIONS)
    assert any(f.code == "BAG_MISSING_LASER_SCAN" for f in findings)


def test_empty_bag_is_critical():
    findings = check_topics(BagInfo(), LIO_SAM_EXPECTATIONS)
    assert findings[0].code == "BAG_EMPTY"
    assert findings[0].severity == Severity.CRITICAL


# ---------------------------------------------------------------- frame ids
def test_frame_id_check_passes_valid_frames():
    findings = check_frame_ids(load_json(GOOD_JSON))
    assert findings[0].code == "BAG_FRAME_IDS_OK"


def test_frame_id_check_catches_an_empty_frame():
    info = load_json(GOOD_JSON)
    info.topics[0].frame_id = ""
    findings = check_frame_ids(info)
    assert any(f.code == "BAG_EMPTY_FRAME_ID" and f.severity == Severity.CRITICAL
               for f in findings)


def test_frame_id_check_catches_a_ros1_leading_slash():
    info = load_json(GOOD_JSON)
    info.topics[0].frame_id = "/velodyne"
    findings = check_frame_ids(info)
    assert any(f.code == "BAG_TF1_FRAME_ID" for f in findings)


def test_frame_id_check_catches_a_mismatch_with_the_config():
    info = load_json(GOOD_JSON)
    findings = check_frame_ids(info, expected={"/points": "os_lidar"})
    assert any(f.code == "BAG_UNEXPECTED_FRAME_ID" for f in findings)


# ----------------------------------------------------------------------- TF
def test_tf_graph_records_parents_and_children():
    g = build_tf_graph([("a", "b", True), ("b", "c", False)])
    assert g["parents"]["b"] == ["a"]
    assert g["children"]["b"] == ["c"]
    assert g["frames"] == {"a", "b", "c"}


def test_tf_check_accepts_a_valid_tree():
    edges = [(e["parent"], e["child"], e["static"]) for e in GOOD_JSON["tf"]]
    findings = check_tf_completeness(
        edges, [("base_link", "velodyne"), ("base_link", "imu_link")])
    assert all(f.severity == Severity.OK for f in findings)
    assert len(findings) == 2


def test_tf_check_catches_a_missing_chain():
    edges = [("base_link", "velodyne", True), ("odom", "base_link", False)]
    findings = check_tf_completeness(edges, [("base_link", "imu_link")])
    bad = [f for f in findings if f.code == "TF_CHAIN_MISSING"]
    assert bad and bad[0].severity == Severity.CRITICAL
    assert "static_transform_publisher" in bad[0].fix


def test_tf_check_catches_two_parents():
    edges = [("odom", "base_link", False), ("map", "base_link", False)]
    findings = check_tf_completeness(edges)
    bad = [f for f in findings if f.code == "TF_MULTIPLE_PARENTS"]
    assert bad and bad[0].severity == Severity.CRITICAL
    assert sorted(bad[0].data["parents"]) == ["map", "odom"]


def test_tf_check_catches_a_static_dynamic_conflict():
    edges = [("base_link", "velodyne", True), ("base_link", "velodyne", False)]
    findings = check_tf_completeness(edges)
    assert any(f.code == "TF_STATIC_DYNAMIC_CONFLICT" for f in findings)


def test_tf_check_catches_disconnected_subtrees():
    edges = [("base_link", "velodyne", True), ("other_root", "imu_link", True)]
    findings = check_tf_completeness(edges)
    roots = [f for f in findings if f.code == "TF_MULTIPLE_ROOTS"]
    assert roots and sorted(roots[0].data["roots"]) == ["base_link", "other_root"]


def test_tf_check_traverses_edges_in_either_direction():
    """tf2 can look up a child->parent transform, so the graph is undirected."""
    edges = [("base_link", "velodyne", True), ("base_link", "imu_link", True)]
    findings = check_tf_completeness(edges, [("velodyne", "imu_link")])
    assert findings[0].code == "TF_CHAIN_OK"


def test_empty_tf_is_critical():
    findings = check_tf_completeness([], [("a", "b")])
    assert findings[0].code == "TF_EMPTY"


# ------------------------------------------------------------------ end to end
def test_check_bag_ranks_everything_together():
    data = json.loads(json.dumps(GOOD_JSON))
    data["tf"] = [e for e in data["tf"] if e["child"] != "imu_link"]
    rep = check_bag(load_json(data), LIO_SAM_EXPECTATIONS,
                    required_chains=[("base_link", "velodyne"),
                                     ("base_link", "imu_link")])
    assert rep.worst == Severity.CRITICAL
    assert rep.problems[0].code == "TF_CHAIN_MISSING"
    assert rep.has("BAG_OK_POINT_CLOUD")


def test_check_bag_warns_about_unknown_duration():
    info = BagInfo(topics=[TopicInfo("/points", "sensor_msgs/msg/PointCloud2", 10)])
    rep = check_bag(info, LIO_SAM_EXPECTATIONS)
    assert rep.has("BAG_ZERO_DURATION")


def test_report_serialises_to_json():
    rep = check_bag(load_json(GOOD_JSON), LIO_SAM_EXPECTATIONS,
                    required_chains=[("base_link", "imu_link")])
    text = json.dumps(rep.to_dict())
    assert "findings" in text
