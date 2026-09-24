from __future__ import annotations

import logging

import pytest

from agentic_kit.domain.models import TurnRequest
from agentic_kit.engine.guardrails.detectors import Detector, default_detectors
from agentic_kit.engine.guardrails.findings import (
    Action,
    Checkpoint,
    Finding,
    GuardrailContext,
    Level,
    Mode,
    Verdict,
)
from agentic_kit.engine.guardrails.guard import Guardrails
from agentic_kit.engine.guardrails.profile import (
    DEFAULT_PROFILE,
    CheckpointPolicy,
    GuardrailProfile,
)
from agentic_kit.engine.guardrails.responder import HANDOFF_REPLY, GuardrailResponder

REQUEST = TurnRequest(conversation_id="conv-1", customer_id="cust-1", text="hi")


class RecordingDetector:
    """Always reports at the given level, and remembers whether it was asked."""

    def __init__(self, detector_id: str, level: Level, action: Action = Action.BLOCK) -> None:
        self.id = detector_id
        self.checkpoints = frozenset({Checkpoint.INPUT, Checkpoint.OUTPUT})
        self.calls = 0
        self._level = level
        self._action = action

    def evaluate(self, context: GuardrailContext) -> Finding | None:
        self.calls += 1
        return Finding(
            detector_id=self.id,
            checkpoint=context.checkpoint,
            level=self._level,
            message=f"{self.id} spoke up",
            action=self._action,
        )


class SilentDetector:
    def __init__(self, detector_id: str = "silent") -> None:
        self.id = detector_id
        self.checkpoints = frozenset({Checkpoint.INPUT})
        self.calls = 0

    def evaluate(self, context: GuardrailContext) -> Finding | None:
        self.calls += 1
        return None


def profile_with(mode: Mode, *detector_ids: str) -> GuardrailProfile:
    return GuardrailProfile(
        checkpoints={Checkpoint.INPUT: CheckpointPolicy(mode=mode, detector_ids=detector_ids)},
        rules=(),
    )


def check(guardrails: Guardrails, text: str = "anything") -> Verdict:
    return guardrails.check(Checkpoint.INPUT, REQUEST, text)


def test_disabled_checkpoint_never_runs_a_detector() -> None:
    detector = RecordingDetector("boom", Level.FAIL)
    guardrails = Guardrails([detector], profile_with(Mode.DISABLED, "boom"))

    assert check(guardrails).passed is True
    assert detector.calls == 0


def test_checkpoint_missing_from_the_profile_is_treated_as_disabled() -> None:
    detector = RecordingDetector("boom", Level.FAIL)
    guardrails = Guardrails([detector], GuardrailProfile(checkpoints={}))

    assert check(guardrails).passed is True
    assert detector.calls == 0


def test_enforce_stops_at_the_first_failure() -> None:
    first = RecordingDetector("first", Level.FAIL)
    second = RecordingDetector("second", Level.FAIL)
    guardrails = Guardrails([first, second], profile_with(Mode.ENFORCE, "first", "second"))

    verdict = check(guardrails)

    assert verdict.passed is False
    assert verdict.failure is not None
    assert verdict.failure.detector_id == "first"
    assert second.calls == 0


def test_enforce_keeps_going_past_a_warning() -> None:
    warned = RecordingDetector("warned", Level.WARN)
    failed = RecordingDetector("failed", Level.FAIL)
    guardrails = Guardrails([warned, failed], profile_with(Mode.ENFORCE, "warned", "failed"))

    verdict = check(guardrails)

    assert verdict.passed is False
    assert [f.detector_id for f in verdict.findings] == ["warned", "failed"]


def test_observe_runs_every_detector_and_still_passes() -> None:
    first = RecordingDetector("first", Level.FAIL)
    second = RecordingDetector("second", Level.FAIL)
    guardrails = Guardrails([first, second], profile_with(Mode.OBSERVE, "first", "second"))

    verdict = check(guardrails)

    assert verdict.passed is True
    assert len(verdict.findings) == 2
    assert second.calls == 1


def test_only_detectors_named_by_the_profile_run() -> None:
    listed = SilentDetector("listed")
    unlisted = RecordingDetector("unlisted", Level.FAIL)
    guardrails = Guardrails([listed, unlisted], profile_with(Mode.ENFORCE, "listed"))

    assert check(guardrails).passed is True
    assert listed.calls == 1
    assert unlisted.calls == 0


def test_a_detector_that_does_not_serve_the_checkpoint_is_skipped() -> None:
    output_only = RecordingDetector("output_only", Level.FAIL)
    output_only.checkpoints = frozenset({Checkpoint.OUTPUT})
    guardrails = Guardrails([output_only], profile_with(Mode.ENFORCE, "output_only"))

    assert check(guardrails).passed is True
    assert output_only.calls == 0


def test_unknown_detector_id_is_logged_and_ignored(caplog: pytest.LogCaptureFixture) -> None:
    guardrails = Guardrails([], profile_with(Mode.ENFORCE, "does_not_exist"))

    with caplog.at_level(logging.WARNING, logger="agentic_kit.guardrails"):
        assert check(guardrails).passed is True

    assert "does_not_exist" in caplog.text


def test_findings_are_logged(caplog: pytest.LogCaptureFixture) -> None:
    guardrails = Guardrails(
        [RecordingDetector("noisy", Level.FAIL)], profile_with(Mode.ENFORCE, "noisy")
    )

    with caplog.at_level(logging.INFO, logger="agentic_kit.guardrails"):
        check(guardrails)

    assert "noisy spoke up" in caplog.text


def test_clean_text_passes_with_no_findings() -> None:
    guardrails = Guardrails([SilentDetector()], profile_with(Mode.ENFORCE, "silent"))

    verdict = check(guardrails)

    assert verdict.passed is True
    assert verdict.findings == ()
    assert verdict.failure is None


def test_rules_come_from_the_profile() -> None:
    assert Guardrails(default_detectors(), DEFAULT_PROFILE).rules == DEFAULT_PROFILE.rules


def test_default_profile_names_only_detectors_that_exist() -> None:
    known = {detector.id for detector in default_detectors()}

    for policy in DEFAULT_PROFILE.checkpoints.values():
        assert set(policy.detector_ids) <= known


def test_handoff_findings_get_the_handoff_reply() -> None:
    finding = Finding("policy_phrase", Checkpoint.INPUT, Level.FAIL, "x", Action.HANDOFF)

    assert GuardrailResponder().reply_for(finding) == HANDOFF_REPLY


@pytest.mark.parametrize("detector_id", ["prompt_injection", "input_length", "something_new"])
def test_every_input_block_has_a_reply(detector_id: str) -> None:
    finding = Finding(detector_id, Checkpoint.INPUT, Level.FAIL, "x")

    assert GuardrailResponder().reply_for(finding)


@pytest.mark.parametrize("checkpoint", list(Checkpoint))
def test_every_checkpoint_has_a_fallback_reply(checkpoint: Checkpoint) -> None:
    finding = Finding("unknown_detector", checkpoint, Level.FAIL, "x")

    assert GuardrailResponder().reply_for(finding)


@pytest.mark.parametrize("detector", default_detectors(), ids=lambda d: d.id)
def test_default_detectors_pass_an_ordinary_message(detector: Detector) -> None:
    for checkpoint in detector.checkpoints:
        context = GuardrailContext(checkpoint, REQUEST, "Where is my order 1001?")
        assert detector.evaluate(context) is None
