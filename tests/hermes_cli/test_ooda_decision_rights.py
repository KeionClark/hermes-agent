"""Effect-based OODA decision rights, with no LLM or clock dependency."""

from dataclasses import FrozenInstanceError, replace

import pytest

from hermes_cli.ooda_decision_rights import (
    ActionContext, ActionType, DecisionOutcome, ReservedEffect, classify,
)


FULL = ActionContext(
    evidence_ref="case/source/2026-09-30", effect_scope="practice A pilot",
    audience="existing authorized staff", reserved_effects=frozenset(),
    same_purpose=True, bounded_pilot=True, bounded_safe_scope=True,
    send_cap=1, qa_revision="reviewed sha", project_ruling_ref="ruling 1",
    release_gate_ref="gate 1", external_access_ready=True,
    opt_in=True, dependency_ready=True, rollback_ready=True,
    production_witness_ref="witness 1",
)


@pytest.mark.parametrize("action,expected", [
    (ActionType.UNBLOCK_TRANSIENT, DecisionOutcome.AGENT_AUTHORIZED),
    (ActionType.RUN_TESTS, DecisionOutcome.AGENT_AUTHORIZED),
    (ActionType.CREATE_INVESTIGATION_CARD, DecisionOutcome.AGENT_AUTHORIZED),
    (ActionType.UPDATE_SKILL, DecisionOutcome.AGENT_AUTHORIZED),
    (ActionType.WRITE_MEMORY, DecisionOutcome.AGENT_AUTHORIZED),
    (ActionType.GRADE_CONVERSATION, DecisionOutcome.AGENT_AUTHORIZED),
    (ActionType.ADJUST_CRON_SCHEDULE, DecisionOutcome.DARK_BUILD_ONLY),
    (ActionType.CREATE_BUILD_CARD, DecisionOutcome.AGENT_AUTHORIZED),
    (ActionType.REMEASURE_METRIC, DecisionOutcome.AGENT_AUTHORIZED),
    (ActionType.ARCHIVE_STALE_PROPOSAL, DecisionOutcome.AGENT_AUTHORIZED),
    (ActionType.UNPAUSE_CRON, DecisionOutcome.DARK_BUILD_ONLY),
    (ActionType.MERGE_STAGING, DecisionOutcome.DARK_BUILD_ONLY),
    (ActionType.ADJUST_THRESHOLD, DecisionOutcome.DARK_BUILD_ONLY),
    (ActionType.CREATE_OODA_AGENT, DecisionOutcome.DARK_BUILD_ONLY),
    (ActionType.OPEN_PR, DecisionOutcome.AGENT_AUTHORIZED),
    (ActionType.MODIFY_WEBHOOK_ROUTES, DecisionOutcome.DARK_BUILD_ONLY),
    (ActionType.SEND_OPERATIONAL_MESSAGE, DecisionOutcome.DARK_BUILD_ONLY),
    (ActionType.CHANGE_CONVERSATION_FLOW, DecisionOutcome.DARK_BUILD_ONLY),
    (ActionType.BULK_SEND, DecisionOutcome.KEION_DECISION),
    (ActionType.CHANGE_BOOKING_RULE, DecisionOutcome.DARK_BUILD_ONLY),
    (ActionType.CHANGE_PATIENT_DATA_HANDLING, DecisionOutcome.DARK_BUILD_ONLY),
    (ActionType.INCREASE_SPEND, DecisionOutcome.KEION_DECISION),
    (ActionType.PIVOT_STRATEGY, DecisionOutcome.KEION_DECISION),
    (ActionType.EXTERNAL_PRODUCT_CHANGE, DecisionOutcome.DARK_BUILD_ONLY),
    (ActionType.DEPLOY_PRODUCTION, DecisionOutcome.DARK_BUILD_ONLY),
])
def test_each_architecture_action_has_policy_route(action, expected):
    assert classify(action, FULL).outcome is expected


@pytest.mark.parametrize("action", list(ActionType))
@pytest.mark.parametrize("effect", list(ReservedEffect))
def test_reserved_effect_overrides_every_action_and_complete_packet(action, effect):
    result = classify(action, replace(FULL, reserved_effects=frozenset({effect})))
    assert result.outcome is DecisionOutcome.KEION_DECISION
    assert result.owner == "keion"


@pytest.mark.parametrize("action", list(ActionType))
def test_silence_is_never_permission(action):
    outcome = classify(action, ActionContext()).outcome
    assert outcome in {DecisionOutcome.EVIDENCE_NEEDED, DecisionOutcome.KEION_DECISION}
    assert outcome is not DecisionOutcome.AGENT_AUTHORIZED


@pytest.mark.parametrize("change", [
    {"evidence_ref": " "}, {"effect_scope": None}, {"audience": ""},
    {"same_purpose": None}, {"same_purpose": False},
    {"bounded_pilot": None}, {"bounded_pilot": False},
    {"bounded_safe_scope": None}, {"bounded_safe_scope": False},
    {"dependency_ready": None},
])
def test_internal_action_requires_source_and_bounded_effect(change):
    assert classify(ActionType.CREATE_INVESTIGATION_CARD, replace(FULL, **change)).outcome is DecisionOutcome.EVIDENCE_NEEDED


def test_dependency_wait_is_explicit_not_a_timeout():
    assert classify(ActionType.CREATE_INVESTIGATION_CARD, replace(FULL, dependency_ready=False)).outcome is DecisionOutcome.DEPENDENCY_WAIT


@pytest.mark.parametrize("action", [
    ActionType.ADJUST_CRON_SCHEDULE, ActionType.UNPAUSE_CRON,
    ActionType.MODIFY_WEBHOOK_ROUTES, ActionType.CREATE_OODA_AGENT,
])
def test_infrastructure_changes_require_explicit_opt_in(action):
    for value in (None, False):
        assert classify(action, replace(FULL, opt_in=value)).outcome is DecisionOutcome.EVIDENCE_NEEDED


@pytest.mark.parametrize("change", [
    {"qa_revision": None}, {"project_ruling_ref": " "},
    {"release_gate_ref": None}, {"rollback_ready": None},
    {"external_access_ready": None},
])
def test_live_change_cannot_default_proceed_on_partial_packet(change):
    assert classify(ActionType.DEPLOY_PRODUCTION, replace(FULL, **change)).outcome is DecisionOutcome.EVIDENCE_NEEDED


def test_access_failure_requires_external_access_not_ceo():
    decision = classify(ActionType.DEPLOY_PRODUCTION, replace(FULL, external_access_ready=False))
    assert decision.outcome is DecisionOutcome.EXTERNAL_ACCESS_NEEDED
    assert decision.owner == "release_engineer"


@pytest.mark.parametrize("send_cap", [None, 0, -1, True, "1"])
def test_send_action_requires_positive_numeric_cap(send_cap):
    assert classify(ActionType.CHANGE_CONVERSATION_FLOW, replace(FULL, send_cap=send_cap)).outcome is DecisionOutcome.EVIDENCE_NEEDED


def test_negative_control_same_purpose_data_fix_is_not_blanket_ceo_gate():
    assert classify(ActionType.CHANGE_PATIENT_DATA_HANDLING, FULL).outcome is DecisionOutcome.DARK_BUILD_ONLY
    assert classify(ActionType.CHANGE_PATIENT_DATA_HANDLING, replace(FULL, reserved_effects=frozenset({ReservedEffect.NEW_DATA_SHARING_EXPORT_ACCESS_OR_PURPOSE}))).outcome is DecisionOutcome.KEION_DECISION


def test_approval_reference_and_witness_do_not_override_reserved_effect():
    decision = classify(ActionType.BULK_SEND, replace(FULL, explicit_approval_ref="unverified unrelated approval"))
    assert decision.outcome is DecisionOutcome.KEION_DECISION


def test_missing_reserved_effect_attestation_does_not_authorize():
    assert classify(ActionType.RUN_TESTS, replace(FULL, reserved_effects=None)).outcome is DecisionOutcome.EVIDENCE_NEEDED


def test_untyped_inputs_and_unknown_effects_fail_closed():
    assert classify("run_tests", FULL).outcome is DecisionOutcome.EVIDENCE_NEEDED
    assert classify(ActionType.RUN_TESTS, None).outcome is DecisionOutcome.EVIDENCE_NEEDED
    assert classify(ActionType.RUN_TESTS, replace(FULL, reserved_effects=frozenset({"unknown"}))).outcome is DecisionOutcome.EVIDENCE_NEEDED


def test_context_and_decision_are_immutable():
    with pytest.raises(FrozenInstanceError):
        FULL.audience = "new patients"
    with pytest.raises(FrozenInstanceError):
        classify(ActionType.RUN_TESTS, FULL).owner = "keion"
