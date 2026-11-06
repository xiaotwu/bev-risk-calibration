# Longitudinal risk and stale-observation correction

This CPU companion evaluates the paper's constant-velocity reference and fixed age correction on a supplied front-object observation history. It includes the current-coordinate constant-velocity tracker, frozen age-bin calibration factors, conditional Gaussian contact integral, and action selection. Install NumPy and SciPy and run the following from the repository root.

```bash
python -m risk
python -m unittest risk.test_risk
```

The synthetic example passes the same input to both policies. After a two-second hold, it prints the two gap and speed means, candidate model risks in action-set order, and the selected accelerations. The example illustrates the command rule and is separate from the archived experimental scenes.

## Input contract

`LegacyCVTracker.update(observation, ego_x, ego_speed, pending_acc, time)` accepts increasing controller timestamps in seconds, longitudinal ego position in meters, ego speed in meters per second, and a committed acceleration in meters per second squared. Use `None` for an unavailable packet and a dictionary with `front=None` for an available packet with no front detection. A positive packet provides acquisition time, an identity such as `frame`, and a front detection with a three-dimensional center in current ego coordinates and absolute velocity components aligned with that frame. The first component of each vector is longitudinal. The score is a finite value in [0, 1]. See the example for concrete field names.

`MotionPolicy` uses the returned tracker record, including the timestamp of the most recent delivered positive detection. Set `age_stress=True` to enable the fixed correction and use the same risk threshold and calibration in the two arms. The paper's closed-loop comparison used a threshold of 0.01. The action set is [-6, -3, 0, 1.5, 3] meters per second squared. The integrator evaluates the model probability under the fixed hypothetical braking contract. The probability depends on the assumed belief and is separate from observed collision frequency.

## Provenance and scope

The integration and control modules are unchanged research implementations. The portable tracker and policy extract the CV-only methods from the archived V4/V5 sources and use local imports in place of the original simulator and runtime loading. The calibration file retains only the fields consumed by this policy from the frozen fit on 256 calibration histories. `source_sha256.json` records the original input digests.

The package starts at the supplied observation interface. Recreating sensor frames and closed-loop simulation requires the original simulator setup and detector. The numerical checks validate the portable code path on deterministic inputs.
