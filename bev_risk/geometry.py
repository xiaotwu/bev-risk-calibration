"""Shared, annotation-independent BEV corridor geometry."""
import numpy as np


def corridor_gap(footprint_ego, config):
    """Clip an ordered convex footprint to the lateral corridor, then take rear x.

    Preserve longitudinal overlaps with the virtual front plane as negative gaps.
    Footprints fully behind that plane or beyond range are ineligible.
    """
    polygon=[np.asarray(p,dtype=float)[:2] for p in footprint_ego]
    half=config["corridor_half_width_m"]
    for bound,sign in ((-half,1.),(half,-1.)):
        if not polygon: return None
        clipped=[]
        for previous,current in zip(polygon[-1:]+polygon[:-1],polygon):
            p_inside=sign*(previous[1]-bound)>=0
            c_inside=sign*(current[1]-bound)>=0
            if p_inside != c_inside:
                ratio=(bound-previous[1])/(current[1]-previous[1])
                clipped.append(previous+ratio*(current-previous))
            if c_inside: clipped.append(current)
        polygon=clipped
    if not polygon: return None
    xs=np.array(polygon)[:,0];front=config["ego_front_offset_m"]
    gap=float(xs.min()-front)
    if xs.max()<front or gap>config["range_m"]: return None
    return gap
