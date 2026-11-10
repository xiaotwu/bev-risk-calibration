"""Shared observation-state fallback and action-sensitive evidence admission."""
import copy
import numpy as np
from bev_risk.geometry import corridor_gap
from bev_risk.policy import ACTIONS,admission,choose_action,nominal_action,stopping_distance


def candidate_on_path(candidate,config):
    x0,y0,x1,y1=candidate['footprint_ego_xyxy']
    return corridor_gap([[x0,y0],[x0,y1],[x1,y1],[x1,y0]],config) is not None


def candidate_masks(t,candidates,pose,ego_speed,past,previous_high_tracker,config):
    """All gate decisions precede mutation of the actual mode trackers.

    R2 single-candidate counterfactuals use a common high-threshold history and
    fixed age allowance, independent of the downstream factorial condition.
    """
    high=[c for c in candidates if (c.get('camera_score') or 0)>=.5]
    baseline=copy.deepcopy(previous_high_tracker);baseline.update(t,high,pose)
    state=baseline.lead(t,pose,config)
    control=lambda s:choose_action(s['gap_m'] if s else None,s['lead_speed_mps'] if s else None,ego_speed,
                                  s['age_s'] if s else 0.,config['gate_age_allowance_s'],0.)
    base_action=control(state)['action']
    masks={k:[] for k in ('L','H','S','R0','C','R2')};records=[]
    for i,(candidate,history) in enumerate(zip(candidates,past)):
        score=candidate.get('camera_score') or 0.;on_path=candidate_on_path(candidate,config)
        high_score=score>=.5;low=.2<=score<.5
        r0,margin=admission(candidate,'R',ego_speed,history['speed_mps'],config['ego_front_offset_m'])
        hypothetical=None;selected=False
        if low and on_path:
            hypothetical_tracker=copy.deepcopy(previous_high_tracker)
            # Preserve original candidate order for deterministic association.
            inserted=[c for j,c in enumerate(candidates) if (c.get('camera_score') or 0)>=.5 or j==i]
            hypothetical_tracker.update(t,inserted,pose)
            hypothetical_state=hypothetical_tracker.lead(t,pose,config)
            selected=bool(hypothetical_state and hypothetical_state['last_candidate_id']==candidate['candidate_id'])
            hypothetical=control(hypothetical_state)['action']
        action_relevant=selected and hypothetical!=base_action
        accepted=dict(L=True,H=high_score,S=score>=.2,R0=r0,C=high_score or (low and on_path),R2=high_score or (low and on_path and action_relevant))
        for mode in masks:masks[mode].append(bool(accepted[mode]))
        records.append(dict(candidate_id=candidate['candidate_id'],camera_score=score,on_path=on_path,
                            past_velocity=history,legacy_margin_m=margin,high_reference_action=base_action,
                            hypothetical_action=hypothetical,candidate_selected_in_hypothetical=selected,
                            action_relevant=bool(action_relevant),accepted=accepted))
    quota=sum(keep and .2<=(c.get('camera_score') or 0)<.5 for c,keep in zip(candidates,masks['R2']))
    eligible=[i for i,c in enumerate(candidates) if .2<=(c.get('camera_score') or 0)<.5 and records[i]['on_path']]
    ranked=sorted(eligible,key=lambda i:(-(candidates[i].get('camera_score') or 0),candidates[i]['candidate_id']))
    chosen=set(ranked[:quota]);masks['K']=[(c.get('camera_score') or 0)>=.5 or i in chosen for i,c in enumerate(candidates)]
    for i,record in enumerate(records):record['accepted']['K']=bool(masks['K'][i]);record['r2_low_quota']=quota
    return masks,records


def decide(state,ego_speed,evidence,allowance_s=None,buffer_m=0.,repair=False,free_guard_m=1.):
    if state is not None:
        decision=choose_action(state['gap_m'],state['lead_speed_mps'],ego_speed,state['age_s'],allowance_s,buffer_m)
        return decision|dict(observation_state='tracked',free_space_used=False)
    distance=evidence['free_distance_m']
    valid=bool(np.isfinite(distance) and distance>=0 and evidence['current_ground_valid'])
    admissible=[a for a in ACTIONS if a<=nominal_action(ego_speed) and distance>stopping_distance(ego_speed,a)+free_guard_m+buffer_m] if valid else []
    if repair and valid and admissible:
        return dict(action=max(admissible),nominal_action=nominal_action(ego_speed),reason='finite_observed_free_prefix',
                    corrected_gap_m=None,corrected_speed_mps=None,support_m=None,age_active=False,
                    correction_nonzero=False,observation_state='evidence_free',free_space_used=True,
                    free_boundary_m=distance,free_guard_m=free_guard_m)
    return choose_action(None,None,ego_speed)|dict(observation_state='unknown',free_space_used=False,correction_nonzero=False)
