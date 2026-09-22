"""Unit tests for check_and_train.py's threshold decision logic."""

from check_and_train import should_attempt, INITIAL_THRESHOLD, RETRAIN_INCREMENT


def test_below_initial_threshold_does_not_attempt():
    attempt, reason = should_attempt(current=INITIAL_THRESHOLD - 1, state={})
    assert attempt is False
    assert "INITIAL_THRESHOLD" in reason


def test_first_attempt_at_threshold_proceeds():
    attempt, _ = should_attempt(current=INITIAL_THRESHOLD, state={"last_attempted_count": 0})
    assert attempt is True


def test_above_threshold_but_no_new_data_since_last_attempt_waits():
    state = {"last_attempted_count": INITIAL_THRESHOLD + 10}
    attempt, reason = should_attempt(current=INITIAL_THRESHOLD + 10 + (RETRAIN_INCREMENT - 1), state=state)
    assert attempt is False
    assert "new example" in reason


def test_enough_new_data_since_last_attempt_proceeds():
    state = {"last_attempted_count": INITIAL_THRESHOLD + 10}
    attempt, _ = should_attempt(current=INITIAL_THRESHOLD + 10 + RETRAIN_INCREMENT, state=state)
    assert attempt is True


def test_force_bypasses_all_gates():
    attempt, reason = should_attempt(current=0, state={}, force=True)
    assert attempt is True
    assert reason == "forced"
