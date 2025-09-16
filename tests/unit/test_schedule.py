import pytest


def test_schedule_basic_and_curve():
    from column_popper.core.schedule import Schedule

    # Start with interval 2.0s, then after 5s switch to 1.0s
    sched = Schedule(game_duration=10.0, initial_interval=2.0, curve=[(5.0, 1.0)])

    # First 5 steps at dt=1 → should produce falls at t=2,4
    falls_total = 0
    for _ in range(5):
        falls_total += sched.advance_step(dt=1.0)
    assert falls_total == 2
    assert pytest.approx(sched.time_left, rel=1e-6) == 5.0

    # Interval now 1.0, next 5 steps should produce 5 falls
    falls_total = 0
    for _ in range(5):
        falls_total += sched.advance_step(dt=1.0)
    assert falls_total == 5
    assert sched.truncated is True

