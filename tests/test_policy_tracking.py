import unittest
import numpy as np
from bev_risk.policy import stopping_distance, stress_shift, choose_action, admission
from bev_risk.tracking import CausalTracker

CFG=dict(ego_front_offset_m=2.5,corridor_half_width_m=1.75,range_m=60.)
def candidate(x,score=.3):
    return dict(center_world=[x,0,1],center_ego=[x,0,1],footprint_ego_xyxy=[x-1,-1,x+1,1],
                observed_rear_x_ego_m=x-1,camera_score=score)

class PolicyTrackingTests(unittest.TestCase):
    def test_stopping_at_low_speed_does_not_reverse(self):
        self.assertAlmostEqual(stopping_distance(.6,-6),.12+.03)
        self.assertEqual(stopping_distance(0,-6),0)
    def test_age_clips_speed_but_accumulates_cv_gap(self):
        self.assertEqual(stress_shift(100,3,3.2,.2),(92.5,0.))
        self.assertEqual(stress_shift(100,3,.1,.2),(100,3))
    def test_strict_boundary_and_missing(self):
        self.assertEqual(choose_action(None,None,10)["action"],-6)
        distance=stopping_distance(10,3)
        self.assertLess(choose_action(distance,0,10)["action"],3)
        self.assertEqual(choose_action(distance+.01,0,10)["action"],3)
    def test_risk_retention_is_selective(self):
        self.assertTrue(admission(candidate(10),"R",15,0)[0])
        self.assertFalse(admission(candidate(60),"R",5,0)[0])
        self.assertTrue(admission(candidate(60),"S",5,0)[0])
    def test_past_only_velocity_and_missing_update_age(self):
        t=CausalTracker(); pose=np.eye(4)
        self.assertFalse(t.past_speeds(0,[candidate(10)],pose)[0]["reliable"])
        t.update(0,[candidate(10)],pose)
        t.update(100000000,[candidate(11)],pose)
        self.assertAlmostEqual(t.past_speeds(200000000,[candidate(12)],pose)[0]["speed_mps"],10)
        with self.assertRaises(ValueError): t.past_speeds(100000000,[candidate(11)],pose)
        t.update(200000000,[],pose)
        lead=t.lead(200000000,pose,CFG)
        self.assertAlmostEqual(lead["age_s"],.1)
        self.assertAlmostEqual(lead["center_world"][0],12)
        self.assertFalse(lead["updated"])
    def test_ego_motion_does_not_create_object_velocity(self):
        t=CausalTracker(); pose=np.eye(4); first=candidate(10)
        t.update(0,[first],pose)
        pose[0,3]=1; second=candidate(9); second["center_world"]=[10,0,1]
        t.update(100000000,[second],pose)
        self.assertAlmostEqual(t.lead(100000000,pose,CFG)["lead_speed_mps"],0)

if __name__ == "__main__": unittest.main()
