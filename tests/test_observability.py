import unittest
import numpy as np
from bev_risk_v2.observability import CorridorEvidence
from bev_risk_v2.method import decide,candidate_masks
from bev_risk.tracking import CausalTracker

CFG=dict(ego_front_offset_m=2.5,corridor_half_width_m=1.75,range_m=60.,gate_age_allowance_s=.2)
def points():
    return np.array([[x,y,0.] for x in np.arange(2.6,12.5,.2) for y in np.arange(-1.7,1.75,.2)])
def cand(x,score):
    return dict(candidate_id='candidate',center_world=[x,0,1],center_ego=[x,0,1],
                footprint_ego_xyxy=[x-1,-1,x+1,1],observed_rear_x_ego_m=x-1,camera_score=score)

class ObservationTests(unittest.TestCase):
    def test_no_returns_are_unknown(self):
        r=CorridorEvidence().update(0,np.eye(4),np.zeros((0,3)),[0,0,0],1.)
        self.assertEqual(r['free_distance_m'],0.)
        self.assertEqual(decide(None,5,r,repair=True)['observation_state'],'unknown')
    def test_supported_prefix_can_release_fallback(self):
        r=CorridorEvidence().update(0,np.eye(4),points(),[0,0,0],1.)
        self.assertGreaterEqual(r['free_distance_m'],8.)
        self.assertEqual(decide(None,5,r,repair=True)['observation_state'],'evidence_free')
        self.assertEqual(decide(None,5,r,repair=False)['action'],-6)
    def test_obstacle_and_missing_rows_interrupt(self):
        obstacle=np.array([[6,.1,1],[6.1,.1,1],[6.2,.1,1]])
        r=CorridorEvidence().update(0,np.eye(4),np.vstack([points(),obstacle]),[0,0,0],1.)
        self.assertEqual(r['free_distance_m'],0.)
    def test_evidence_expires_and_bad_current_plane_blocks(self):
        m=CorridorEvidence();m.update(0,np.eye(4),points(),[0,0,0],1.)
        r=m.update(400000000,np.eye(4),np.zeros((0,3)),[0,0,0],1.)
        self.assertEqual(r['free_distance_m'],0.)
        r=m.update(500000000,np.eye(4),points(),[0,0,0],.2)
        self.assertEqual(r['free_distance_m'],0.)
    def test_action_rescue_skips_saturated_candidate(self):
        tracker=CausalTracker();pose=np.eye(4)
        histories=[dict(speed_mps=0.,reliable=False,history_age_s=None,source_track_id=None)]
        masks,records=candidate_masks(0,[cand(7,.3)],pose,15,histories,tracker,CFG)
        self.assertFalse(masks['R2'][0]);self.assertTrue(masks['R0'][0])
        masks,records=candidate_masks(0,[cand(30,.3)],pose,5,histories,tracker,CFG)
        self.assertTrue(masks['R2'][0]);self.assertFalse(masks['R0'][0])
    def test_high_evidence_is_preserved(self):
        masks,_=candidate_masks(0,[cand(7,.9)],np.eye(4),15,[dict(speed_mps=0.)],CausalTracker(),CFG)
        self.assertTrue(masks['R2'][0])

if __name__=='__main__':unittest.main()
