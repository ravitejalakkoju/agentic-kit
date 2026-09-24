"""Runs the detectors a checkpoint asks for and reports one verdict."""

from __future__ import annotations

import logging
from collections.abc import Iterable

from ...domain.models import TurnRequest
from .detectors import Detector
from .findings import Checkpoint, Finding, GuardrailContext, Level, Mode, Verdict
from .profile import GuardrailProfile

logger = logging.getLogger("agentic_kit.guardrails")

PASSED = Verdict(passed=True)


class Guardrails:
    def __init__(self, detectors: Iterable[Detector], profile: GuardrailProfile) -> None:
        self._detectors = {detector.id: detector for detector in detectors}
        self._profile = profile

    @property
    def rules(self) -> tuple[str, ...]:
        return self._profile.rules

    def check(
        self, checkpoint: Checkpoint, text: str, *, request: TurnRequest | None = None
    ) -> Verdict:
        """Run the checkpoint's detectors over one piece of text.

        Ingesting a document is a checkpoint with no turn behind it, so the
        request is optional and only ever used for the log line.
        """
        policy = self._profile.policy_for(checkpoint)
        if policy.mode is Mode.DISABLED:
            return PASSED

        enforcing = policy.mode is Mode.ENFORCE
        context = GuardrailContext(checkpoint=checkpoint, text=text, request=request)
        findings: list[Finding] = []

        for detector in self._selected(checkpoint, policy.detector_ids):
            finding = detector.evaluate(context)
            if finding is None:
                continue
            self._log(request, finding)
            findings.append(finding)
            if enforcing and finding.level is Level.FAIL:
                break

        failed = enforcing and any(f.level is Level.FAIL for f in findings)
        return Verdict(passed=not failed, findings=tuple(findings))

    def _selected(self, checkpoint: Checkpoint, detector_ids: Iterable[str]) -> Iterable[Detector]:
        for detector_id in detector_ids:
            detector = self._detectors.get(detector_id)
            if detector is None:
                logger.warning("profile names unknown detector %s", detector_id)
            elif checkpoint in detector.checkpoints:
                yield detector

    def _log(self, request: TurnRequest | None, finding: Finding) -> None:
        logger.info(
            "guardrail %s at %s: %s",
            finding.detector_id,
            finding.checkpoint,
            finding.message,
            extra={
                "conversation_id": request.conversation_id if request else None,
                "level": finding.level,
            },
        )
