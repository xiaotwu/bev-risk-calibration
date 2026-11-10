"""Causal shared/accepted-track histories; this module never reads annotations."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linear_sum_assignment
from bev_risk.geometry import corridor_gap


@dataclass
class Track:
    track_id: int
    timestamp_ns: int
    center_world: np.ndarray
    footprint_world: np.ndarray
    velocity_world: np.ndarray
    observations: int = 1
    last_candidate_id: str = ""

    def predict(self, timestamp_ns):
        dt = (timestamp_ns-self.timestamp_ns)*1e-9
        if dt < 0:
            raise ValueError("Future track read")
        return self.center_world+self.velocity_world*dt, self.footprint_world+self.velocity_world*dt


def footprint_world(candidate, world_T_ego):
    x0, y0, x1, y1 = candidate["footprint_ego_xyxy"]
    z = candidate["center_ego"][2]
    points = np.array([[x0,y0,z],[x0,y1,z],[x1,y1,z],[x1,y0,z]])
    return points @ world_T_ego[:3,:3].T + world_T_ego[:3,3]


class CausalTracker:
    def __init__(self, max_age_s=3., association_base_m=3., association_growth_mps=2.,
                 association_max_m=5., velocity_alpha=.5, max_speed_mps=35.):
        self.max_age_s = max_age_s
        self.association_base_m = association_base_m
        self.association_growth_mps = association_growth_mps
        self.association_max_m = association_max_m
        self.velocity_alpha = velocity_alpha
        self.max_speed_mps = max_speed_mps
        self.tracks = {}
        self.next_id = 0
        self.last_step_ns = None

    def associate(self, timestamp_ns, candidates):
        tracks = [t for t in self.tracks.values() if 0 <= (timestamp_ns-t.timestamp_ns)*1e-9 <= self.max_age_s]
        if not tracks or not candidates:
            return {}
        predicted = np.array([t.predict(timestamp_ns)[0][:2] for t in tracks])
        centers = np.array([c["center_world"][:2] for c in candidates])
        distances = np.linalg.norm(predicted[:,None,:]-centers[None,:,:],axis=2)
        limits = np.array([min(self.association_max_m,self.association_base_m+self.association_growth_mps*(timestamp_ns-t.timestamp_ns)*1e-9) for t in tracks])
        gated = np.where(distances <= limits[:,None],distances,1e6)
        rows,cols = linear_sum_assignment(gated)
        return {int(c):tracks[int(r)] for r,c in zip(rows,cols) if gated[r,c] < 1e6}

    def past_speeds(self, timestamp_ns, candidates, world_T_ego):
        """Read association and velocity BEFORE incorporating the current sweep."""
        if self.last_step_ns is not None and timestamp_ns <= self.last_step_ns:
            raise ValueError("Pre-admission criticality requires strictly past history")
        matches = self.associate(timestamp_ns,candidates)
        rows = []
        for i in range(len(candidates)):
            t = matches.get(i)
            reliable = t is not None and t.observations >= 2
            rows.append(dict(speed_mps=max(0.,float((world_T_ego[:3,:3].T@t.velocity_world)[0])) if reliable else 0.,
                             reliable=reliable, history_age_s=(timestamp_ns-t.timestamp_ns)*1e-9 if t else None,
                             source_track_id=t.track_id if t else None))
        return rows

    def update(self, timestamp_ns, candidates, world_T_ego):
        if self.last_step_ns is not None and timestamp_ns <= self.last_step_ns:
            raise ValueError("Tracker input must be strictly increasing")
        self.tracks = {k:t for k,t in self.tracks.items() if (timestamp_ns-t.timestamp_ns)*1e-9 <= self.max_age_s}
        matches = self.associate(timestamp_ns,candidates)
        for i,candidate in enumerate(candidates):
            center = np.asarray(candidate["center_world"],dtype=float)
            footprint = footprint_world(candidate,world_T_ego)
            t = matches.get(i)
            if t is None:
                t = Track(self.next_id,timestamp_ns,center,footprint,np.zeros(3))
                self.tracks[self.next_id] = t
                self.next_id += 1
            else:
                dt = (timestamp_ns-t.timestamp_ns)*1e-9
                measured = (center-t.center_world)/dt
                measured[2] = 0.
                norm = np.linalg.norm(measured[:2])
                if norm > self.max_speed_mps:
                    measured *= self.max_speed_mps/norm
                t.velocity_world = measured if t.observations == 1 else self.velocity_alpha*measured+(1.-self.velocity_alpha)*t.velocity_world
                t.center_world,t.footprint_world,t.timestamp_ns = center,footprint,timestamp_ns
                t.observations += 1
            t.last_candidate_id = str(candidate.get("candidate_id",i))
        self.last_step_ns = timestamp_ns

    def lead(self, timestamp_ns, world_T_ego, config):
        inverse = np.linalg.inv(world_T_ego)
        choices = []
        for t in self.tracks.values():
            age = (timestamp_ns-t.timestamp_ns)*1e-9
            if age > self.max_age_s:
                continue
            center, footprint = t.predict(timestamp_ns)
            ego = footprint@inverse[:3,:3].T+inverse[:3,3]
            gap = corridor_gap(ego,config)
            if gap is None:
                continue
            choices.append(dict(track_id=t.track_id,center_world=center.tolist(),footprint_world=footprint.tolist(),
                                gap_m=gap,lead_speed_mps=max(0.,float((inverse[:3,:3]@t.velocity_world)[0])),
                                age_s=age,initialized=True,updated=t.timestamp_ns==timestamp_ns,
                                last_update_ns=t.timestamp_ns,observations=t.observations,last_candidate_id=t.last_candidate_id))
        return min(choices,key=lambda s:(s["gap_m"],s["track_id"])) if choices else None
