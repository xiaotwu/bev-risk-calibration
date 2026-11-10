"""Independent conservative-behavior checks; synthetic data only, no holdout IO."""
import pickle
import unittest
import numpy as np
from bev_risk.tracking import CausalTracker
from bev_risk.policy import nominal_action,stopping_distance
from bev_risk_v2.method import candidate_masks,decide
from bev_risk_v2.observability import CorridorEvidence

CFG=dict(ego_front_offset_m=2.5,corridor_half_width_m=1.75,range_m=60.,gate_age_allowance_s=.2)

def candidate(identity,x,score,y=0.):
    return dict(candidate_id=identity,center_world=[x,y,1.],center_ego=[x,y,1.],
                footprint_ego_xyxy=[x-1,y-1,x+1,y+1],observed_rear_x_ego_m=x-1,camera_score=score)

def histories(candidates):
    return [dict(speed_mps=0.,reliable=False,history_age_s=None,source_track_id=None) for _ in candidates]

def ground_points():
    return np.array([[x,y,0.] for x in np.arange(2.6,12.5,.2) for y in np.arange(-1.7,1.75,.2)])

class IndependentMethodReviewTests(unittest.TestCase):
    def test_k_equal_low_quota_but_ranks_camera_score_not_r2_choice(self):
        candidates=[candidate('high_side',20,.9,y=8),candidate('near_highlow',6,.49),
                    candidate('near_next',6,.45),candidate('far1',30,.21),candidate('far2',35,.4)]
        masks,records=candidate_masks(0,candidates,np.eye(4),5,histories(candidates),CausalTracker(),CFG)
        self.assertEqual(masks['R2'],[True,False,False,True,True])
        self.assertEqual(masks['K'],[True,True,True,False,False])
        self.assertEqual(sum(masks['R2']),sum(masks['K']))
        self.assertTrue(all(r['r2_low_quota']==2 for r in records))

    def test_k_ties_use_identity_and_zero_quota_retains_high_only(self):
        candidates=[candidate('b',6,.49),candidate('a',6,.49),candidate('rescue',30,.21)]
        masks,_=candidate_masks(0,candidates,np.eye(4),5,histories(candidates),CausalTracker(),CFG)
        self.assertEqual(masks['K'],[False,True,False])
        high=[candidate('high',7,.5),candidate('saturated_low',6,.3)]
        masks,_=candidate_masks(0,high,np.eye(4),15,histories(high),CausalTracker(),CFG)
        self.assertEqual(masks['K'],masks['H'])

    def test_all_high_scores_survive_path_and_action_filters(self):
        candidates=[candidate('high_near',6,.5),candidate('high_side',20,.9,y=9),candidate('below',6,.199)]
        masks,_=candidate_masks(0,candidates,np.eye(4),15,histories(candidates),CausalTracker(),CFG)
        for mode in ('H','S','R0','C','R2','K'):
            self.assertEqual(masks[mode][:2],[True,True]);self.assertFalse(masks[mode][2])

    def test_counterfactuals_do_not_mutate_previous_high_history(self):
        tracker=CausalTracker();pose=np.eye(4)
        tracker.update(0,[candidate('old0',20,.8)],pose)
        tracker.update(100_000_000,[candidate('old1',20.1,.8)],pose)
        before=pickle.dumps(tracker,protocol=5)
        candidates=[candidate('newhigh',20.2,.8),candidate('low',8,.3)]
        candidate_masks(200_000_000,candidates,pose,5,histories(candidates),tracker,CFG)
        self.assertEqual(before,pickle.dumps(tracker,protocol=5))
        self.assertEqual(tracker.last_step_ns,100_000_000)

    def test_free_boundary_uses_strict_inequality_and_nominal_cap(self):
        speed=5.;boundary=stopping_distance(speed,3.)+1.
        evidence=dict(free_distance_m=boundary,current_ground_valid=True)
        d=decide(None,speed,evidence,repair=True)
        self.assertLess(d['action'],3.)
        self.assertGreater(boundary,stopping_distance(speed,d['action'])+1.)
        highspeed=30.;large=dict(free_distance_m=10_000.,current_ground_valid=True)
        self.assertLessEqual(decide(None,highspeed,large,repair=True)['action'],nominal_action(highspeed))
        self.assertEqual(decide(None,speed,evidence,repair=False)['action'],-6.)

    def test_invalid_free_distance_cannot_release_unknown_fallback(self):
        for distance in (float('inf'),float('nan'),-1.):
            d=decide(None,5.,dict(free_distance_m=distance,current_ground_valid=True),repair=True)
            self.assertEqual(d['observation_state'],'unknown');self.assertEqual(d['action'],-6.)
        d=decide(None,5.,dict(free_distance_m=100.,current_ground_valid=False),repair=True)
        self.assertEqual(d['action'],-6.)

    def test_tracked_state_takes_precedence_over_ground_free_evidence(self):
        state=dict(gap_m=0.,lead_speed_mps=0.,age_s=0.)
        d=decide(state,10.,dict(free_distance_m=100.,current_ground_valid=True),repair=True)
        self.assertEqual(d['observation_state'],'tracked');self.assertEqual(d['action'],-6.)
        self.assertFalse(d['free_space_used'])

    def test_nonfinite_current_plane_cannot_borrow_old_ground_coverage(self):
        for plane in ([0.,0.,float('nan')],[0.,0.]):
            evidence=CorridorEvidence();evidence.update(0,np.eye(4),ground_points(),[0,0,0],1.)
            current=evidence.update(100_000_000,np.eye(4),ground_points(),plane,1.)
            self.assertFalse(current['current_ground_valid']);self.assertEqual(current['free_distance_m'],0.)

    def test_inclusive_300ms_window_uses_integer_time(self):
        evidence=CorridorEvidence();evidence.update(0,np.eye(4),ground_points(),[0,0,0],1.)
        boundary=evidence.update(300_000_000,np.eye(4),np.empty((0,3)),[0,0,0],1.)
        self.assertGreater(boundary['free_distance_m'],0.)
        expired=evidence.update(300_000_001,np.eye(4),np.empty((0,3)),[0,0,0],1.)
        self.assertEqual(expired['free_distance_m'],0.)

    def test_occupancy_dilation_uses_external_lateral_and_front_halo(self):
        # These returns are just outside the corridor/front plane, but their
        # frozen one-cell uncertainty dilation intersects its first row.
        for obstacle in (np.array([[6.,1.80,1.],[6.1,1.80,1.],[6.2,1.80,1.]]),
                         np.array([[2.1,.1,1.],[2.2,.1,1.],[2.3,.1,1.]])):
            r=CorridorEvidence().update(0,np.eye(4),np.vstack([ground_points(),obstacle]),[0,0,0],1.)
            self.assertEqual(r['free_distance_m'],0.)

if __name__=='__main__':unittest.main()
