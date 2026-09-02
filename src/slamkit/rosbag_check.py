"""Bag sanity checks: topics, rates, frame_ids, TF completeness.

The first thing to do with a bag is not to run SLAM on it.  It is to check
that the bag contains what the SLAM node is going to ask for.  Roughly half of
"LIO-SAM does not start" is a topic name, a QoS mismatch, or a missing static
transform, and all three are visible from ``ros2 bag info`` plus one look at
the TF tree.

Guarded imports
---------------
``rosbag2_py`` is used when it is importable.  When it is not -- on a laptop,
in CI, on the machine you actually do the analysis on -- everything still
works from a JSON or text dump the user can produce in one command:

.. code-block:: console

   $ ros2 bag info my_bag > baginfo.txt
   $ python3 -m slamkit.rosbag_check baginfo.txt

The text parser understands the standard ``ros2 bag info`` output.  For frame
ids and TF you need a little more, so :func:`load_json` accepts an extended
schema that also carries ``frame_ids`` and ``tf`` (see :data:`JSON_SCHEMA_DOC`).
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .findings import Finding, Report, Severity

__all__ = [
    "HAVE_ROSBAG2",
    "TopicInfo",
    "BagInfo",
    "TopicExpectation",
    "LIO_SAM_EXPECTATIONS",
    "CARTOGRAPHER_2D_EXPECTATIONS",
    "RTABMAP_EXPECTATIONS",
    "parse_ros2_bag_info_text",
    "load_json",
    "load_bag",
    "load_any",
    "check_topics",
    "check_frame_ids",
    "build_tf_graph",
    "check_tf_completeness",
    "check_bag",
    "JSON_SCHEMA_DOC",
]

try:  # pragma: no cover - depends on the host having ROS 2
    import rosbag2_py  # type: ignore

    HAVE_ROSBAG2 = True
except Exception:  # pragma: no cover
    rosbag2_py = None  # type: ignore
    HAVE_ROSBAG2 = False


JSON_SCHEMA_DOC = """\
{
  "duration_s": 123.4,
  "message_count": 123456,
  "topics": [
    {"name": "/points", "type": "sensor_msgs/msg/PointCloud2", "count": 1234},
    {"name": "/imu/data", "type": "sensor_msgs/msg/Imu", "count": 24000}
  ],
  "frame_ids": {"/points": "velodyne", "/imu/data": "imu_link"},
  "tf": [
    {"parent": "base_link", "child": "velodyne", "static": true},
    {"parent": "base_link", "child": "imu_link", "static": true},
    {"parent": "odom",      "child": "base_link", "static": false}
  ]
}
"""


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------
@dataclass
class TopicInfo:
    """One topic in a bag."""

    name: str
    type: str = ""
    count: int = 0
    frequency_hz: float = float("nan")
    frame_id: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class BagInfo:
    """What we know about a bag, however we learned it."""

    path: str = ""
    duration_s: float = 0.0
    message_count: int = 0
    topics: List[TopicInfo] = field(default_factory=list)
    tf_edges: List[Tuple[str, str, bool]] = field(default_factory=list)
    """``(parent, child, is_static)`` triples."""
    source: str = "unknown"
    """``rosbag2``, ``json``, or ``text``."""

    def topic(self, name: str) -> Optional[TopicInfo]:
        for t in self.topics:
            if t.name == name:
                return t
        return None

    def find(self, pattern: str) -> List[TopicInfo]:
        """Topics whose name or type matches a regex."""
        rx = re.compile(pattern)
        return [t for t in self.topics if rx.search(t.name) or rx.search(t.type)]

    def to_dict(self) -> Dict[str, object]:
        return {
            "path": self.path,
            "duration_s": self.duration_s,
            "message_count": self.message_count,
            "source": self.source,
            "topics": [t.to_dict() for t in self.topics],
            "tf_edges": [list(e) for e in self.tf_edges],
        }


@dataclass
class TopicExpectation:
    """What a SLAM stack needs on a topic."""

    role: str
    """Human name, e.g. ``"point cloud"``."""
    type_pattern: str
    """Regex matched against the message type."""
    min_hz: float = 0.0
    required: bool = True
    note: str = ""


LIO_SAM_EXPECTATIONS: List[TopicExpectation] = [
    TopicExpectation("point cloud", r"PointCloud2", min_hz=5.0, required=True,
                     note="pointCloudTopic. Must carry per-point ring and time fields."),
    TopicExpectation("IMU", r"sensor_msgs/msg/Imu|sensor_msgs/Imu", min_hz=150.0,
                     required=True,
                     note="imuTopic. LIO-SAM wants >= 200 Hz and a 9-axis IMU if you "
                          "use useImuHeadingInitialization."),
    TopicExpectation("GPS odometry", r"nav_msgs/msg/Odometry|nav_msgs/Odometry",
                     min_hz=0.0, required=False,
                     note="gpsTopic, optional; only used when useGpsElevation or GPS "
                          "factors are enabled."),
]

CARTOGRAPHER_2D_EXPECTATIONS: List[TopicExpectation] = [
    TopicExpectation("laser scan", r"LaserScan", min_hz=5.0, required=True,
                     note="num_laser_scans = 1 in the .lua."),
    TopicExpectation("IMU", r"sensor_msgs/msg/Imu|sensor_msgs/Imu", min_hz=100.0,
                     required=False,
                     note="Optional in 2D, mandatory in 3D. If present you MUST set "
                          "use_imu_data = true and tracking_frame to the IMU frame."),
    TopicExpectation("odometry", r"nav_msgs/msg/Odometry|nav_msgs/Odometry",
                     min_hz=10.0, required=False,
                     note="use_odometry = true. Strongly recommended in corridors."),
]

RTABMAP_EXPECTATIONS: List[TopicExpectation] = [
    TopicExpectation("point cloud or scan", r"PointCloud2|LaserScan", min_hz=1.0,
                     required=True, note="subscribe_scan_cloud / subscribe_scan."),
    TopicExpectation("odometry", r"nav_msgs/msg/Odometry|nav_msgs/Odometry",
                     min_hz=5.0, required=True,
                     note="RTAB-Map needs an odometry source; icp_odometry or "
                          "rgbd_odometry can provide it if the robot does not."),
    TopicExpectation("camera image", r"sensor_msgs/msg/Image|CompressedImage",
                     min_hz=1.0, required=False,
                     note="Needed for visual loop closure (Kp/ features). Without it, "
                          "loop closure is ICP-only and much weaker."),
]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
_TOPIC_LINE = re.compile(
    r"Topic:\s*(?P<name>\S+)\s*\|\s*Type:\s*(?P<type>\S+)\s*\|\s*Count:\s*(?P<count>\d+)"
)
_DURATION = re.compile(r"^\s*Duration:\s*([0-9.]+)s", re.MULTILINE)
_MESSAGES = re.compile(r"^\s*Messages:\s*(\d+)", re.MULTILINE)


def parse_ros2_bag_info_text(text: str, path: str = "") -> BagInfo:
    """Parse the human-readable output of ``ros2 bag info``.

    This exists so a user with no Python on the robot can send you one command's
    output and get a full diagnosis back.
    """
    info = BagInfo(path=path, source="text")
    m = _DURATION.search(text)
    if m:
        info.duration_s = float(m.group(1))
    m = _MESSAGES.search(text)
    if m:
        info.message_count = int(m.group(1))
    for tm in _TOPIC_LINE.finditer(text):
        count = int(tm.group("count"))
        freq = count / info.duration_s if info.duration_s > 0 else float("nan")
        info.topics.append(TopicInfo(name=tm.group("name"), type=tm.group("type"),
                                     count=count, frequency_hz=freq))
    if not info.topics:
        raise ValueError(
            "no 'Topic: ... | Type: ... | Count: ...' lines found; this does not look "
            "like `ros2 bag info` output"
        )
    return info


def load_json(source) -> BagInfo:
    """Load a bag description from a dict, a JSON string, or a path to a file.

    See :data:`JSON_SCHEMA_DOC` for the schema.  Unknown keys are ignored, so
    you can hand it a richer dump without stripping it first.
    """
    if isinstance(source, dict):
        data = source
        path = str(source.get("path", ""))
    elif isinstance(source, (str, os.PathLike)) and os.path.exists(str(source)):
        path = str(source)
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    elif isinstance(source, str):
        data = json.loads(source)
        path = str(data.get("path", ""))
    else:
        raise TypeError(f"cannot load bag info from {type(source).__name__}")
    info = BagInfo(path=path, source="json")
    info.duration_s = float(data.get("duration_s", data.get("duration", 0.0)) or 0.0)
    info.message_count = int(data.get("message_count", data.get("messages", 0)) or 0)
    frame_ids = data.get("frame_ids", {}) or {}
    for t in data.get("topics", []) or []:
        name = t.get("name") or t.get("topic") or ""
        count = int(t.get("count", t.get("message_count", 0)) or 0)
        freq = t.get("frequency_hz")
        if freq is None:
            freq = count / info.duration_s if info.duration_s > 0 else float("nan")
        info.topics.append(TopicInfo(
            name=name,
            type=str(t.get("type", "")),
            count=count,
            frequency_hz=float(freq),
            frame_id=t.get("frame_id", frame_ids.get(name)),
        ))
    for e in data.get("tf", []) or []:
        info.tf_edges.append(
            (str(e.get("parent", "")), str(e.get("child", "")), bool(e.get("static", False)))
        )
    return info


def load_bag(path: str) -> BagInfo:  # pragma: no cover - requires rosbag2
    """Read a rosbag2 directory with ``rosbag2_py``. Raises if it is unavailable."""
    if not HAVE_ROSBAG2:
        raise RuntimeError(
            "rosbag2_py is not importable here. Run "
            "`ros2 bag info <bag> > baginfo.txt` on the robot and pass that file "
            "instead -- slamkit parses it directly."
        )
    reader = rosbag2_py.SequentialReader()
    storage = rosbag2_py.StorageOptions(uri=path, storage_id="")
    converter = rosbag2_py.ConverterOptions("", "")
    reader.open(storage, converter)
    meta = reader.get_metadata()
    duration = meta.duration.nanoseconds / 1e9 if hasattr(meta, "duration") else 0.0
    info = BagInfo(path=path, source="rosbag2", duration_s=duration,
                   message_count=int(getattr(meta, "message_count", 0)))
    for t in meta.topics_with_message_count:
        count = int(t.message_count)
        info.topics.append(TopicInfo(
            name=t.topic_metadata.name,
            type=t.topic_metadata.type,
            count=count,
            frequency_hz=count / duration if duration > 0 else float("nan"),
        ))
    return info


def load_any(source) -> BagInfo:
    """Best-effort load: rosbag2 directory, JSON dump, or ``ros2 bag info`` text."""
    if isinstance(source, dict):
        return load_json(source)
    path = str(source)
    if os.path.isdir(path) and HAVE_ROSBAG2:  # pragma: no cover
        return load_bag(path)
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as fh:
            raw = fh.read()
        stripped = raw.lstrip()
        if stripped.startswith("{"):
            return load_json(json.loads(raw))
        return parse_ros2_bag_info_text(raw, path=path)
    raise FileNotFoundError(f"{path!r} is not a bag directory, JSON dump or info text")


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------
def check_topics(info: BagInfo,
                 expectations: Sequence[TopicExpectation] = tuple(LIO_SAM_EXPECTATIONS),
                 ) -> List[Finding]:
    """Check that the bag has the topics the stack needs, at a usable rate."""
    out: List[Finding] = []
    if not info.topics:
        out.append(Finding(
            code="BAG_EMPTY",
            severity=Severity.CRITICAL,
            message="the bag lists no topics",
            symptom="SLAM node starts and sits there. `ros2 topic hz` shows nothing.",
            fix="Recording failed. Check disk space and that you recorded the right "
                "topics -- `ros2 bag record -a` if in doubt.",
        ))
        return out
    for exp in expectations:
        matches = [t for t in info.topics if re.search(exp.type_pattern, t.type)]
        if not matches:
            out.append(Finding(
                code=f"BAG_MISSING_{exp.role.upper().replace(' ', '_')}",
                severity=Severity.CRITICAL if exp.required else Severity.INFO,
                message=f"no topic of type matching /{exp.type_pattern}/ "
                        f"({exp.role}) in the bag",
                symptom="The SLAM node subscribes and never receives a callback; no "
                        "error is printed because subscribing to a nonexistent topic "
                        "is legal in ROS 2." if exp.required else "",
                fix=(f"{exp.note} Check the topic name in your config against "
                     "`ros2 topic list`, and remember that a remap in a launch file "
                     "overrides the parameter.") if exp.required else exp.note,
            ))
            continue
        best = max(matches, key=lambda t: t.count)
        if exp.min_hz > 0 and math.isfinite(best.frequency_hz) and best.frequency_hz < exp.min_hz:
            out.append(Finding(
                code=f"BAG_LOW_RATE_{exp.role.upper().replace(' ', '_')}",
                severity=Severity.ERROR,
                message=f"{best.name} ({exp.role}) averages "
                        f"{best.frequency_hz:.1f} Hz, below the {exp.min_hz:.0f} Hz "
                        f"this stack expects",
                symptom="Under-constrained motion between updates: the estimator "
                        "extrapolates and the map lags, worst during turns.",
                fix=exp.note + " Also check for dropped messages: a topic can average "
                               "a low rate because of gaps rather than a low nominal "
                               "rate. Compare `ros2 topic hz` at the driver with the "
                               "count in the bag.",
                data=best.to_dict(),
            ))
        else:
            out.append(Finding(
                code=f"BAG_OK_{exp.role.upper().replace(' ', '_')}",
                severity=Severity.OK,
                message=f"{best.name} ({exp.role}, {best.type}) at "
                        f"{best.frequency_hz:.1f} Hz, {best.count} messages",
                data=best.to_dict(),
            ))
    return out


def check_frame_ids(info: BagInfo, expected: Optional[Dict[str, str]] = None
                    ) -> List[Finding]:
    """Check that sensor messages carry a frame_id and that it is the expected one."""
    out: List[Finding] = []
    known = [t for t in info.topics if t.frame_id is not None]
    if not known:
        out.append(Finding(
            code="BAG_NO_FRAME_INFO",
            severity=Severity.INFO,
            message="no frame_id information supplied; add a 'frame_ids' map to the "
                    "JSON dump to enable this check",
            fix="On the robot: `ros2 topic echo --once /points | head -5` shows the "
                "frame_id in the header.",
        ))
        return out
    for t in known:
        if not t.frame_id:
            out.append(Finding(
                code="BAG_EMPTY_FRAME_ID",
                severity=Severity.CRITICAL,
                message=f"{t.name} publishes an empty header.frame_id",
                symptom="RViz shows 'For frame []: Frame [] does not exist'. The SLAM "
                        "node cannot look up any transform for this data and either "
                        "drops it or assumes identity, which silently disables your "
                        "extrinsic.",
                fix="Set the driver's frame_id parameter. Every sensor message must "
                    "name the frame its data is in -- that is the entire mechanism by "
                    "which the extrinsic is applied.",
            ))
        elif t.frame_id.startswith("/"):
            out.append(Finding(
                code="BAG_TF1_FRAME_ID",
                severity=Severity.ERROR,
                message=f"{t.name} uses the ROS 1 leading-slash frame '{t.frame_id}'",
                symptom="tf2 lookups fail with 'Invalid argument passed to canTransform'.",
                fix=f"Strip the leading slash: '{t.frame_id.lstrip('/')}'.",
            ))
    if expected:
        for topic, want in expected.items():
            t = info.topic(topic)
            if t is None or t.frame_id is None:
                continue
            if t.frame_id != want:
                out.append(Finding(
                    code="BAG_UNEXPECTED_FRAME_ID",
                    severity=Severity.WARN,
                    message=f"{topic} is stamped '{t.frame_id}', config expects '{want}'",
                    symptom="The extrinsic is looked up between the wrong pair of "
                            "frames, so it is either identity or a completely "
                            "different transform.",
                    fix="Make the driver's frame_id and the SLAM config's "
                        "lidarFrame/imu frame agree, or add the missing static "
                        "transform between them.",
                ))
    if not out:
        out.append(Finding(code="BAG_FRAME_IDS_OK", severity=Severity.OK,
                           message=f"{len(known)} topics carry a valid frame_id"))
    return out


def build_tf_graph(edges: Sequence[Tuple[str, str, bool]]) -> Dict[str, object]:
    """Build parent/child maps from ``(parent, child, static)`` edges."""
    parents: Dict[str, List[str]] = {}
    children: Dict[str, List[str]] = {}
    static: Dict[Tuple[str, str], List[bool]] = {}
    frames: Set[str] = set()
    for parent, child, is_static in edges:
        frames.add(parent)
        frames.add(child)
        parents.setdefault(child, []).append(parent)
        children.setdefault(parent, []).append(child)
        static.setdefault((parent, child), []).append(bool(is_static))
    return {"parents": parents, "children": children, "frames": frames, "static": static}


def _connected(graph: Dict[str, object], a: str, b: str) -> bool:
    """Undirected reachability -- tf2 can traverse an edge in either direction."""
    adj: Dict[str, Set[str]] = {}
    for child, ps in graph["parents"].items():  # type: ignore[index]
        for p in ps:
            adj.setdefault(child, set()).add(p)
            adj.setdefault(p, set()).add(child)
    if a not in adj or b not in adj:
        return False
    seen = {a}
    stack = [a]
    while stack:
        cur = stack.pop()
        if cur == b:
            return True
        for nxt in adj.get(cur, ()):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return False


def check_tf_completeness(
    edges: Sequence[Tuple[str, str, bool]],
    required_chains: Sequence[Tuple[str, str]] = (),
) -> List[Finding]:
    """Validate a TF tree: single parent, connected, and the chains you need exist.

    ``edges`` are ``(parent, child, is_static)``.  ``required_chains`` are
    ``(from_frame, to_frame)`` pairs the SLAM stack will look up -- typically
    ``("base_link", "<lidar frame>")`` and ``("base_link", "<imu frame>")``.
    """
    out: List[Finding] = []
    if not edges:
        out.append(Finding(
            code="TF_EMPTY",
            severity=Severity.CRITICAL,
            message="no TF edges supplied",
            symptom="Everything downstream fails with 'frame does not exist'.",
            fix="Publish the sensor mounts as static transforms in your launch file. "
                "TF is the first thing to get right -- before time sync, before "
                "extrinsics, before any tuning.",
        ))
        return out
    graph = build_tf_graph(edges)
    parents: Dict[str, List[str]] = graph["parents"]  # type: ignore[assignment]

    for child, ps in parents.items():
        uniq = sorted(set(ps))
        if len(uniq) > 1:
            out.append(Finding(
                code="TF_MULTIPLE_PARENTS",
                severity=Severity.CRITICAL,
                message=f"frame '{child}' has {len(uniq)} parents: {uniq}",
                symptom="TF_OLD_DATA / 'TF_REPEATED_DATA' warnings flooding the console, "
                        "and the robot visibly jitters between two positions in RViz.",
                fix="A TF tree is a tree: exactly one parent per frame. This is almost "
                    "always two nodes publishing the same transform -- e.g. a "
                    "robot_state_publisher and a hand-written static_transform_publisher "
                    "for the same joint, or two SLAM nodes both publishing odom->base_link.",
                data={"child": child, "parents": uniq},
            ))
    static_map: Dict[Tuple[str, str], List[bool]] = graph["static"]  # type: ignore
    for (p, c), flags in static_map.items():
        if len(set(flags)) > 1:
            out.append(Finding(
                code="TF_STATIC_DYNAMIC_CONFLICT",
                severity=Severity.ERROR,
                message=f"'{p}' -> '{c}' is published both as static (/tf_static) and "
                        "dynamic (/tf)",
                symptom="The transform flickers between two values; lookups succeed or "
                        "fail depending on timing, so the bug is intermittent.",
                fix="Pick one. Sensor mounts are static; anything a joint moves is "
                    "dynamic. Note /tf_static is latched, so a stale static transform "
                    "survives a node restart -- restart the whole graph when you change it.",
            ))
    for a, b in required_chains:
        if not _connected(graph, a, b):
            out.append(Finding(
                code="TF_CHAIN_MISSING",
                severity=Severity.CRITICAL,
                message=f"no TF path from '{a}' to '{b}'",
                symptom="'Could not find a connection between [a] and [b] because they "
                        "are not part of the same tree'. The SLAM node blocks or drops "
                        "every message.",
                fix=f"Add the missing link. Minimum viable: "
                    f"`ros2 run tf2_ros static_transform_publisher --x .. --y .. --z .. "
                    f"--roll .. --pitch .. --yaw .. --frame-id {a} --child-frame-id {b}` "
                    "(the modern arg form; the old positional form silently uses a "
                    "different argument order and is a classic source of a wrong mount).",
                data={"from": a, "to": b, "frames": sorted(graph["frames"])},  # type: ignore
            ))
        else:
            out.append(Finding(
                code="TF_CHAIN_OK",
                severity=Severity.OK,
                message=f"TF path '{a}' -> '{b}' exists",
            ))
    roots = [f for f in graph["frames"] if f not in parents]  # type: ignore[operator]
    if len(roots) > 1:
        out.append(Finding(
            code="TF_MULTIPLE_ROOTS",
            severity=Severity.ERROR,
            message=f"the TF tree has {len(roots)} disconnected roots: {sorted(roots)}",
            symptom="Some lookups work and others fail with 'not part of the same tree'.",
            fix="Connect the subtrees. Usually a sensor frame was attached to a frame "
                "name that differs from the URDF by one character.",
            data={"roots": sorted(roots)},
        ))
    return out


def check_bag(
    info: BagInfo,
    expectations: Sequence[TopicExpectation] = tuple(LIO_SAM_EXPECTATIONS),
    required_chains: Sequence[Tuple[str, str]] = (),
    expected_frame_ids: Optional[Dict[str, str]] = None,
) -> Report:
    """Run every bag check and return one ranked :class:`Report`."""
    rep = Report(title=f"Bag check: {info.path or '<in memory>'} (via {info.source})")
    if info.duration_s <= 0:
        rep.add(Finding(
            code="BAG_ZERO_DURATION",
            severity=Severity.WARN,
            message="bag duration is zero or unknown; rate checks are disabled",
            fix="Pass duration_s in the JSON dump, or use `ros2 bag info` text output "
                "which includes it.",
        ))
    rep.extend(check_topics(info, expectations))
    rep.extend(check_frame_ids(info, expected_frame_ids))
    if info.tf_edges or required_chains:
        rep.extend(check_tf_completeness(info.tf_edges, required_chains))
    return rep


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - CLI glue
    """``python3 -m slamkit.rosbag_check <baginfo.txt|dump.json|bag_dir>``."""
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("source", help="bag directory, ros2 bag info text, or JSON dump")
    ap.add_argument("--stack", choices=["lio_sam", "cartographer_2d", "rtabmap"],
                    default="lio_sam")
    ap.add_argument("--json", action="store_true", help="emit JSON")
    args = ap.parse_args(argv)
    table = {"lio_sam": LIO_SAM_EXPECTATIONS,
             "cartographer_2d": CARTOGRAPHER_2D_EXPECTATIONS,
             "rtabmap": RTABMAP_EXPECTATIONS}
    info = load_any(args.source)
    rep = check_bag(info, table[args.stack])
    if args.json:
        print(json.dumps({"bag": info.to_dict(), "report": rep.to_dict()}, indent=2))
    else:
        print(rep)
    return 1 if rep.worst >= Severity.ERROR else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
