"""Run both policies on the same synthetic delivered-observation history."""

import json

from . import LegacyCVTracker, MotionPolicy, load_calibration


def main():
    tracker = LegacyCVTracker()
    packet = {
        "frame": 1,
        "acquisition_time": 0.0,
        "sensor_age_s": 0.0,
        "front": {"center": [73.0, 0.0, 0.0], "velocity": [9.0, 0.0, 0.0], "score": 0.9},
    }
    for tick in range(11):
        time = tick * 0.2
        estimate = tracker.update(packet if tick == 0 else None, 16.0 * time, 16.0, -3.0, time)
    calibration = load_calibration()
    policy = {"tracker": "cv_current", "param": 0.01}
    reference = MotionPolicy(policy, calibration).step(estimate, 16.0, -3.0, time)
    correction = MotionPolicy(dict(policy, age_stress=True), calibration).step(estimate, 16.0, -3.0, time)
    result = {"example": "synthetic observation history", "time_s": time}
    for label, decision in (("cv", reference), ("fixed_age_correction", correction)):
        belief = decision["belief"]
        result[label] = {
            "gap_mean_m": belief["mean_gap"],
            "lead_speed_mean_mps": belief["mean_vlead"],
            "command_mps2": decision["command"],
            "candidate_model_risks": decision["diagnostics"]["candidates"]["risk"].tolist(),
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
