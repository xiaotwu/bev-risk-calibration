"""Exercise the released scoring code with deterministic synthetic trajectories."""
import numpy as np
import pandas as pd

from openacc.replay_openacc import score as score_openacc
from replay import eligibility, match, score


def test_replay():
    ticks = np.arange(-6, 34)
    elapsed = ticks / 10
    frames = []
    for cohort, deceleration in enumerate((3., 0.)):
        speed = 12 - deceleration * np.maximum(elapsed, 0)
        lead = 35 + 12 * elapsed - deceleration / 2 * np.maximum(elapsed, 0)**2
        ego = 10 * elapsed
        gap = lead - 4 - ego
        frames.append(pd.DataFrame(dict(
            global_time=1000 + 100 * ticks, frame_id=ticks + 7,
            vehicle_id=2 * cohort + 1, preceding=2 * cohort + 2, lane_id=1,
            lead_y=lead, lead_length=4., local_y=ego, gap=gap, published_v=speed,
        )))
        track = pd.DataFrame(dict(Speed1=speed, Speed2=10., IVS1=gap))
        label = ('braking', 'stable')[cohort]
        events = score_openacc(track, ticks, 'synthetic',
                               {'ACC': ['example'], 'Distance_setting': ['example']},
                               [(6, label, 0)], {6: -0.9 if cohort == 0 else 0})
        row = next(item for item in events if item['age_s'] == 2)
        expected = [(12.75, 7.3125), (0., -5.4375)][cohort]
        np.testing.assert_allclose([row['cv_B_error_m'], row['stress_B_error_m']],
                                   expected, rtol=0, atol=1e-10)

    frame = pd.concat(frames, ignore_index=True)
    braking, stable, _ = eligibility(frame)
    pairs = match(braking, stable)
    np.testing.assert_array_equal(pairs, [[6, 46]])
    scores = score(frame, pairs, 'synthetic', 'synthetic')
    main = scores[(scores.age_s == 2) & (scores.velocity_source == 'position')]
    np.testing.assert_allclose(main[['cv', 'stress']], [[12.75, 7.3125], [0., -5.4375]],
                               rtol=0, atol=1e-10)
    np.testing.assert_array_equal(scores.loc[scores.age_s == 1, 'cv'],
                                  scores.loc[scores.age_s == 1, 'stress'])


if __name__ == '__main__':
    test_replay()
    print('PASS: synthetic matching, stopping-support error, and one-second correction parity')
