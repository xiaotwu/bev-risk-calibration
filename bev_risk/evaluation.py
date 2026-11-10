"""OFFLINE annotation reference and paired action scoring.

This module must not be imported by online perception, tracking, or admission.
The reference uses exact-time cuboids and only backward object displacement;
missing reference velocity is never silently replaced with zero. All quantities
describe the fixed longitudinal backup model, not actual collision probability.
"""
from __future__ import annotations

from bisect import bisect_right
from pathlib import Path

import numpy as np
import pandas as pd

from .av2_data import transform_from_pose, transform_points
from .geometry import corridor_gap


VEHICLE_CATEGORIES = (
    "REGULAR_VEHICLE", "LARGE_VEHICLE", "BUS", "BOX_TRUCK", "TRUCK",
    "TRUCK_CAB", "VEHICULAR_TRAILER", "ARTICULATED_BUS", "SCHOOL_BUS",
)


def _value(config, name, default):
    return (config or {}).get(name, default)


def _convex_hull(points):
    points = sorted(set(map(tuple, np.asarray(points, dtype=float))))
    if len(points) < 3:
        raise ValueError("A reference footprint must have positive area")

    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

    lower, upper = [], []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = np.asarray(lower[:-1]+upper[:-1], dtype=float)
    if len(hull) < 3:
        raise ValueError("Degenerate reference footprint")
    return hull


def _point_polygon_distance(point, polygon):
    """Euclidean distance to a filled convex footprint, zero inside."""
    point, polygon = np.asarray(point, float), np.asarray(polygon, float)
    edges = np.roll(polygon, -1, axis=0)-polygon
    relative = point-polygon
    crosses = edges[:, 0]*relative[:, 1]-edges[:, 1]*relative[:, 0]
    if np.all(crosses >= -1e-10) or np.all(crosses <= 1e-10):
        return 0.0
    squared = np.sum(edges*edges, axis=1)
    fraction = np.clip(np.sum(relative*edges, axis=1)/squared, 0, 1)
    closest = polygon+fraction[:, None]*edges
    return float(np.linalg.norm(closest-point, axis=1).min())


class ReferenceLog:
    """Offline-only reference loader; construction does not score any frame.

    Config is a flat mapping. Defaults: virtual ego_front_offset_m=2.5,
    corridor_half_width_m=1.75, range_m=60, reference_velocity_min_dt_s=.2,
    reference_velocity_max_dt_s=.5, reference_vehicle_categories as above.
    ``at`` requires an exact annotation timestamp; no future interpolation.
    """
    def __init__(self, project_root, log_id, config=None):
        root = Path(project_root)/"data/av2_sensor/val"/str(log_id)
        annotations = pd.read_feather(root/"annotations.feather")
        poses = pd.read_feather(root/"city_SE3_egovehicle.feather")
        self._initialize(annotations, poses, config, str(log_id))

    @classmethod
    def from_tables(cls, annotations, poses, config=None, log_id="synthetic"):
        """Construct from records/DataFrames for deterministic fixture checks."""
        instance = cls.__new__(cls)
        instance._initialize(pd.DataFrame(annotations), pd.DataFrame(poses), config, log_id)
        return instance

    def _initialize(self, annotations, poses, config, log_id):
        self.config = dict(ego_front_offset_m=2.5, corridor_half_width_m=1.75, range_m=60.)
        self.config.update(config or {})
        self.log_id = str(log_id)
        if self.config["corridor_half_width_m"] <= 0 or self.config["range_m"] <= 0:
            raise ValueError("Positive reference corridor half-width and range required")
        required = {"timestamp_ns", "track_uuid", "category", "length_m", "width_m", "height_m",
                    "qw", "qx", "qy", "qz", "tx_m", "ty_m", "tz_m"}
        if not required.issubset(annotations.columns):
            raise ValueError(f"Missing reference columns: {sorted(required-set(annotations.columns))}")
        if annotations.duplicated(["timestamp_ns", "track_uuid"]).any():
            raise ValueError("Duplicate reference track at one timestamp")
        if poses["timestamp_ns"].duplicated().any():
            raise ValueError("Duplicate reference ego pose")
        self._poses = {int(row["timestamp_ns"]): transform_from_pose(row)
                       for row in poses.to_dict(orient="records")}
        self._annotation_timestamps = set(map(int, annotations["timestamp_ns"]))
        categories = set(_value(self.config, "reference_vehicle_categories", VEHICLE_CATEGORIES))
        if not categories or not categories.issubset(VEHICLE_CATEGORIES):
            raise ValueError("Reference categories must be a nonempty subset of the declared vehicle taxonomy")
        self._frames, self._tracks = {}, {}
        for row in annotations.to_dict(orient="records"):
            timestamp = int(row["timestamp_ns"])
            if timestamp not in self._poses:
                raise ValueError(f"Exact reference ego pose absent at {timestamp}")
            dimensions = np.array([row[k] for k in ("length_m", "width_m", "height_m")], float)
            if not np.isfinite(dimensions).all() or np.any(dimensions <= 0):
                raise ValueError("Invalid reference cuboid dimensions")
            world_T_object = self._poses[timestamp] @ transform_from_pose(row)
            corners = np.array([[x, y, z] for x in (-.5, .5)
                                for y in (-.5, .5) for z in (-.5, .5)])*dimensions
            item = dict(timestamp_ns=timestamp, track_uuid=str(row["track_uuid"]),
                        category=str(row["category"]), center_world=world_T_object[:3, 3],
                        in_scope_vehicle=row["category"] in categories,
                        vertices_world=transform_points(world_T_object, corners))
            self._frames.setdefault(timestamp, []).append(item)
            self._tracks.setdefault(item["track_uuid"], []).append(item)
        for track in self._tracks.values():
            track.sort(key=lambda row: row["timestamp_ns"])
        self._track_times = {key: [row["timestamp_ns"] for row in rows]
                             for key, rows in self._tracks.items()}

    def _velocity(self, item, timestamp, world_T_ego):
        minimum = float(_value(self.config, "reference_velocity_min_dt_s", .2))
        maximum = float(_value(self.config, "reference_velocity_max_dt_s", .5))
        if not 0 < minimum <= maximum:
            raise ValueError("Invalid reference velocity lookback interval")
        times = self._track_times[item["track_uuid"]]
        index = bisect_right(times, timestamp-int(round(minimum*1e9)))-1
        if index < 0:
            return None, None, None
        previous = self._tracks[item["track_uuid"]][index]
        elapsed_ns = timestamp-previous["timestamp_ns"]
        if elapsed_ns <= 0 or elapsed_ns > int(round(maximum*1e9)):
            return None, None, None
        dt = elapsed_ns*1e-9
        velocity_world = (item["center_world"]-previous["center_world"])/dt
        velocity_ego = world_T_ego[:3, :3].T @ velocity_world
        return float(velocity_ego[0]), dt, previous["timestamp_ns"]

    def at(self, timestamp_ns, world_T_ego, ego_speed):
        """Reference dict including all current vehicle footprints for matching.

        The returned object is evaluation-only. Do not pass it to an online arm.
        A timestamp with no annotation rows is flagged missing, not called empty.
        """
        timestamp = int(timestamp_ns)
        transform = np.asarray(world_T_ego, float)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError("Finite 4x4 current ego pose required")
        if not np.isfinite(ego_speed) or ego_speed < 0:
            raise ValueError("Finite nonnegative ego speed required")
        if timestamp not in self._poses or not np.allclose(transform, self._poses[timestamp], atol=1e-7, rtol=1e-7):
            raise ValueError("Supplied current ego pose must equal the exact recorded pose")
        available = timestamp in self._annotation_timestamps
        inverse = np.linalg.inv(transform)
        all_objects = []
        for item in self._frames.get(timestamp, []):
            vertices_ego = transform_points(inverse, item["vertices_world"])
            footprint = _convex_hull(vertices_ego[:, :2])
            gap = corridor_gap(footprint, self.config)
            speed, dt, previous = self._velocity(item, timestamp, transform)
            all_objects.append(dict(track_uuid=item["track_uuid"], category=item["category"],
                                in_scope_vehicle=item["in_scope_vehicle"],
                                center_world=item["center_world"].tolist(),
                                center_ego=transform_points(inverse, item["center_world"][None, :])[0].tolist(),
                                footprint_ego_xy=footprint.tolist(), gap_m=gap,
                                lead_speed_mps=speed, velocity_valid=speed is not None,
                                velocity_lookback_s=dt, velocity_previous_timestamp_ns=previous))
        objects = [row for row in all_objects if row["in_scope_vehicle"]]
        eligible = [row for row in objects if row["gap_m"] is not None]
        eligible.sort(key=lambda row: (row["gap_m"], row["track_uuid"]))
        lead = eligible[0] if eligible else None
        return dict(log_id=self.log_id, timestamp_ns=timestamp, reference_available=available,
                    reference_status=("missing_annotation_timestamp" if not available else
                                      "reference_lead" if lead else "reference_empty"),
                    has_lead=lead is not None, lead=lead, objects=objects, all_objects=all_objects,
                    world_T_ego=transform.tolist(), eligible_vehicle_count=len(eligible),
                    ego_front_offset_is_virtual=True)


def _match(ref, state, config, object_key="objects"):
    if state is None or not state.get("initialized", True):
        return None, None
    # Prefer observed-points center when explicitly supplied; otherwise the
    # track's observed/propagated center, not a GT identity or label.
    center = np.asarray(state.get("observed_center_world", state.get("center_world")), float)
    if center.shape != (3,) or not np.isfinite(center).all():
        footprint = np.asarray(state.get("footprint_world"), float)
        if footprint.ndim != 2 or footprint.shape[1] != 3 or not np.isfinite(footprint).all():
            raise ValueError("Method state requires a finite center or footprint for offline matching")
        center = footprint.mean(axis=0)
    current = transform_points(np.linalg.inv(np.asarray(ref["world_T_ego"])), center[None, :])[0]
    inflation = float(_value(config, "reference_match_inflation_m", .75))
    if inflation < 0:
        raise ValueError("Nonnegative reference match inflation required")
    matches = []
    for obj in ref[object_key]:
        distance = _point_polygon_distance(current[:2], obj["footprint_ego_xy"])
        if distance <= inflation:
            center_distance = float(np.linalg.norm(current[:2]-np.asarray(obj["center_ego"])[:2]))
            matches.append((center_distance, obj["track_uuid"], distance))
    if not matches:
        return None, None
    _, identity, distance = min(matches)
    return identity, distance


def score(ref, state, action, ego_speed, config=None):
    """Score one chosen action without altering/replacing the online state.

    Primary eligibility: reference lead + valid velocity + full-braking margin
    >= -0.10 m. No-reference and model-infeasible frames remain explicit and
    contribute to their respective diagnostics, not the primary denominator.
    ``None`` means undefined/not eligible; callers must not coerce it to zero.
    """
    from .policy import choose_action, nominal_action, stopping_distance

    action, ego_speed = float(action), float(ego_speed)
    if not np.isfinite(action) or action not in (-6., -3., 0., 1.5, 3.):
        raise ValueError("Action must belong to the frozen five-action set")
    if not np.isfinite(ego_speed) or ego_speed < 0:
        raise ValueError("Finite nonnegative ego speed required")
    tolerance = float(_value(config, "reference_deficit_tolerance_m", .10))
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Finite nonnegative reference tolerance required")
    available, has_lead = bool(ref["reference_available"]), bool(ref["has_lead"])
    initialized = state is not None and bool(state.get("initialized", True))
    lead = ref.get("lead")
    velocity_valid = bool(available and has_lead and lead["velocity_valid"])
    match, match_distance = _match(ref, state, config) if available else (None, None)
    any_match, any_distance = _match(ref, state, config, "all_objects") if available else (None, None)
    any_object = next((row for row in ref.get("all_objects", []) if row["track_uuid"] == any_match), None)
    lead_identity = lead["track_uuid"] if has_lead else None
    matched_lead = bool(available and has_lead and initialized and match == lead_identity)
    result = dict(reference_available=available, reference_has_lead=has_lead,
                  reference_empty=bool(available and not has_lead), reference_velocity_valid=velocity_valid,
                  reference_gap_m=float(lead["gap_m"]) if has_lead else None,
                  reference_lead_speed_mps=float(lead["lead_speed_mps"]) if velocity_valid else None,
                  reference_velocity_lookback_s=lead["velocity_lookback_s"] if velocity_valid else None,
                  reference_track_uuid=lead_identity, method_initialized=initialized,
                  matched_reference_lead=matched_lead, matched_gt_track_uuid=match,
                  match_footprint_distance_m=match_distance,
                  matched_any_gt_track_uuid=any_match,
                  matched_any_gt_category=any_object["category"] if any_object is not None else None,
                  match_any_footprint_distance_m=any_distance,
                  out_of_scope_object_match=bool(any_object is not None and not any_object["in_scope_vehicle"]),
                  unmatched_to_any_annotation=bool(available and initialized and any_match is None),
                  no_estimated_lead=bool(available and has_lead and not initialized),
                  wrong_annotated_lead=bool(available and initialized and match is not None and match != lead_identity),
                  unmatched_estimated_lead=bool(available and initialized and match is None),
                  action_mps2=action, reference_action_mps2=None, reference_feasible=None,
                  false_admission_eligible=False, false_admission=None, false_admission_raw=None,
                  reference_margin_m=None, reference_full_brake_margin_m=None,
                  reference_deficit_m=None, extra_deficit_over_full_brake_m=None,
                  overaggressive_action=None, overconservative_action=None,
                  support_signed_error_m=None, support_positive_error_m=None)
    if not available:
        return result
    if not has_lead:
        reference_action = float(nominal_action(ego_speed))
    elif not velocity_valid:
        return result
    else:
        gap, speed = float(lead["gap_m"]), float(lead["lead_speed_mps"])
        d_lead = max(0., speed)**2/16.
        margin = gap-max(0., float(stopping_distance(ego_speed, action))-d_lead)
        brake_margin = gap-max(0., float(stopping_distance(ego_speed, -6.))-d_lead)
        deficit, brake_deficit = max(0., -margin), max(0., -brake_margin)
        feasible = brake_margin >= -tolerance
        reference_action = float(choose_action(gap, speed, ego_speed, age_s=0.,
                                               allowance_s=None, buffer_m=0.)["action"])
        result.update(reference_feasible=feasible, false_admission_eligible=feasible,
                      false_admission=bool(margin < -tolerance) if feasible else None,
                      false_admission_raw=bool(margin < 0.) if brake_margin >= 0. else None,
                      reference_margin_m=margin, reference_full_brake_margin_m=brake_margin,
                      reference_deficit_m=deficit,
                      extra_deficit_over_full_brake_m=max(0., deficit-brake_deficit))
        if matched_lead:
            estimate_gap, estimate_speed = float(state["gap_m"]), float(state["lead_speed_mps"])
            if not np.isfinite([estimate_gap, estimate_speed]).all():
                raise ValueError("Finite method state required for support-error scoring")
            error = estimate_gap+max(0., estimate_speed)**2/16.-(gap+d_lead)
            result.update(support_signed_error_m=float(error), support_positive_error_m=max(0., float(error)))
    result.update(reference_action_mps2=reference_action,
                  overaggressive_action=bool(action > reference_action),
                  overconservative_action=bool(action < reference_action))
    return result


def evaluate_frame(log_ref, timestamp_ns, world_T_ego, ego_speed, method_state, action, config=None):
    """Convenience wrapper; reuse ``at`` + ``score`` to avoid 16 repeated reads."""
    reference = log_ref.at(timestamp_ns, world_T_ego, ego_speed)
    return score(reference, method_state, action, ego_speed,
                 log_ref.config if config is None else config)
