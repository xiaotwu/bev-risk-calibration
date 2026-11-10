"""Causal, conservative observed-ground corridor evidence, without annotations.

This heuristic is not a certified free-space map. Unknown cells interrupt the
free prefix; absence of camera detections alone never supplies free evidence.
"""
from collections import deque
import numpy as np
from scipy.ndimage import binary_dilation

DEFAULTS=dict(history_s=.30,ground_abs_height_m=.18,obstacle_min_height_m=.30,
              obstacle_max_height_m=3.5,min_ground_quality=.50,x_cell_m=2.,y_cell_m=.50,
              min_ground_points=2,min_obstacle_points=3,occupancy_dilation_cells=1,
              quantization_reserve_m=2.,
              ego_front_offset_m=2.5,corridor_half_width_m=1.75,range_m=60.)


class CorridorEvidence:
    def __init__(self,config=None):
        self.config=DEFAULTS|dict(config or {})
        self.history=deque();self.last_time=None

    def update(self,timestamp_ns,world_T_ego,xyz_ego,ground_plane,ground_quality):
        if self.last_time is not None and timestamp_ns<=self.last_time:
            raise ValueError("Strictly increasing evidence times required")
        self.last_time=timestamp_ns;c=self.config
        pose=np.asarray(world_T_ego,float);points=np.asarray(xyz_ego,float)
        if pose.shape!=(4,4) or not np.isfinite(pose).all() or points.ndim!=2 or points.shape[1]!=3:
            raise ValueError("Invalid sensor geometry")
        while self.history and timestamp_ns-self.history[0][0]>round(c['history_s']*1e9):
            self.history.popleft()
        plane=np.asarray(ground_plane,float) if ground_plane is not None else np.array([])
        good=plane.shape==(3,) and np.isfinite(plane).all() and np.isfinite(ground_quality) and ground_quality>=c['min_ground_quality']
        if good:
            p=points[np.isfinite(points).all(axis=1)&(points[:,0]>=2)&(points[:,0]<=65)&(abs(points[:,1])<=3.5)&(abs(points[:,2])<=5)]
            height=p[:,2]-p[:,:2]@plane[:2]-plane[2]
            ground=p[abs(height)<=c['ground_abs_height_m']]
            obstacle=p[(height>=c['obstacle_min_height_m'])&(height<=c['obstacle_max_height_m'])]
            world=lambda pts:pts@pose[:3,:3].T+pose[:3,3]
            self.history.append((timestamp_ns,world(ground),world(obstacle)))
        nx=int(np.ceil(c['range_m']/c['x_cell_m']))
        ny=int(round(2*c['corridor_half_width_m']/c['y_cell_m']))
        halo=int(c['occupancy_dilation_cells'])
        ground_counts=np.zeros((nx,ny),dtype=np.int64);obstacle_counts=np.zeros((nx+2*halo,ny+2*halo),dtype=np.int64)
        inv=np.linalg.inv(pose)
        for _,ground,obstacle in self.history:
            for data,counts in ((ground,ground_counts),(obstacle,obstacle_counts)):
                ego=data@inv[:3,:3].T+inv[:3,3]
                ix=np.floor((ego[:,0]-c['ego_front_offset_m'])/c['x_cell_m']).astype(int)
                iy=np.floor((ego[:,1]+c['corridor_half_width_m'])/c['y_cell_m']).astype(int)
                if counts is obstacle_counts:ix=ix+halo;iy=iy+halo
                valid=(ix>=0)&(ix<counts.shape[0])&(iy>=0)&(iy<counts.shape[1])
                np.add.at(counts,(ix[valid],iy[valid]),1)
        occupied=obstacle_counts>=c['min_obstacle_points']
        if c['occupancy_dilation_cells']:
            occupied=binary_dilation(occupied,structure=np.ones((3,3),bool),iterations=c['occupancy_dilation_cells'])
            occupied=occupied[halo:-halo,halo:-halo]
        supported=(ground_counts>=c['min_ground_points'])&~occupied
        rows=supported.all(axis=1)
        first=int(np.where(~rows)[0][0]) if (~rows).any() else nx
        # A failed current-plane check cannot be bridged by older evidence.
        free_distance=max(0.,min(c['range_m'],first*c['x_cell_m'])-c['quantization_reserve_m']) if good else 0.
        return dict(timestamp_ns=timestamp_ns,free_distance_m=float(free_distance),
                    current_ground_valid=bool(good),ground_quality=float(ground_quality),
                    evidence_oldest_age_s=(timestamp_ns-self.history[0][0])*1e-9 if self.history else None,
                    evidence_sweeps=len(self.history),ground_supported_cells=int((ground_counts>=c['min_ground_points']).sum()),
                    occupied_cells=int(occupied.sum()),total_cells=nx*ny,
                    first_unknown_row=int(first),row_fully_supported=rows.tolist(),
                    state_semantics="observed-ground heuristic, not certified empty space")
