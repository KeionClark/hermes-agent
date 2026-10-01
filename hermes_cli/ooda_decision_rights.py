"""Pure, fail-closed OODA action triage; no action is executed here.

An authorization outcome is a classification, not an approval or release gate.
OPERATING_POLICY.md §29–33 and DECISION_BRAIN.md take precedence over the
older architecture's blanket Tier C labels and Tier B timeout.
"""

from dataclasses import dataclass
from enum import StrEnum


class ActionType(StrEnum):
    UNBLOCK_TRANSIENT = "unblock_transient"
    RUN_TESTS = "run_tests"
    CREATE_INVESTIGATION_CARD = "create_investigation_card"
    UPDATE_SKILL = "update_skill"
    WRITE_MEMORY = "write_memory"
    GRADE_CONVERSATION = "grade_conversation"
    ADJUST_CRON_SCHEDULE = "adjust_cron_schedule"
    CREATE_BUILD_CARD = "create_build_card"
    REMEASURE_METRIC = "remeasure_metric"
    ARCHIVE_STALE_PROPOSAL = "archive_stale_proposal"
    UNPAUSE_CRON = "unpause_cron"
    MERGE_STAGING = "merge_staging"
    ADJUST_THRESHOLD = "adjust_threshold"
    CREATE_OODA_AGENT = "create_ooda_agent"
    OPEN_PR = "open_pr"
    MODIFY_WEBHOOK_ROUTES = "modify_webhook_routes"
    SEND_OPERATIONAL_MESSAGE = "send_operational_message"
    CHANGE_CONVERSATION_FLOW = "change_conversation_flow"
    BULK_SEND = "bulk_send"
    CHANGE_BOOKING_RULE = "change_booking_rule"
    CHANGE_PATIENT_DATA_HANDLING = "change_patient_data_handling"
    INCREASE_SPEND = "increase_spend"
    PIVOT_STRATEGY = "pivot_strategy"
    EXTERNAL_PRODUCT_CHANGE = "external_product_change"
    DEPLOY_PRODUCTION = "deploy_production"


class ReservedEffect(StrEnum):
    NEW_PATIENT_AUDIENCE = "new_patient_audience"
    BULK_OR_ABOVE_CAP_SEND = "bulk_or_above_cap_send"
    INCREASED_CADENCE = "increased_cadence"
    UNAPPROVED_CLINICAL_ADVICE = "unapproved_clinical_advice"
    TREATMENT_OR_PRICE_COMMITMENT = "treatment_or_price_commitment"
    MATERIAL_CONSENT_OR_OPT_OUT = "material_consent_or_opt_out"
    NEW_DATA_SHARING_EXPORT_ACCESS_OR_PURPOSE = "new_data_sharing_export_access_or_purpose"
    IRREVERSIBLE_DELETION_OR_RETENTION_REDUCTION = "irreversible_deletion_or_retention_reduction"
    SPENDING_PURCHASE_BILLING_ACCOUNT_SECURITY_OR_VENDOR_COMMITMENT = "spending_purchase_billing_account_security_or_vendor_commitment"
    MATERIAL_CONFLICT_OR_EXPLICIT_PROJECT_APPROVAL = "material_conflict_or_explicit_project_approval"


class DecisionOutcome(StrEnum):
    AGENT_AUTHORIZED = "agent_authorized"
    DARK_BUILD_ONLY = "dark_build_only"
    EVIDENCE_NEEDED = "evidence_needed"
    EXTERNAL_ACCESS_NEEDED = "external_access_needed"
    KEION_DECISION = "keion_decision"
    DEPENDENCY_WAIT = "dependency_wait"


@dataclass(frozen=True)
class ActionContext:
    evidence_ref: str | None = None
    effect_scope: str | None = None
    audience: str | None = None
    reserved_effects: frozenset[ReservedEffect] | None = None
    same_purpose: bool | None = None
    bounded_pilot: bool | None = None
    send_cap: int | None = None
    qa_revision: str | None = None
    project_ruling_ref: str | None = None
    release_gate_ref: str | None = None
    explicit_approval_ref: str | None = None
    external_access_ready: bool | None = None
    # These are factual attestations, never inferred from a model or elapsed time.
    opt_in: bool | None = None
    bounded_safe_scope: bool | None = None
    dependency_ready: bool | None = None
    rollback_ready: bool | None = None
    production_witness_ref: str | None = None


@dataclass(frozen=True)
class Decision:
    outcome: DecisionOutcome
    reason_code: str
    owner: str


_INTERNAL = frozenset({
    ActionType.UNBLOCK_TRANSIENT, ActionType.RUN_TESTS,
    ActionType.CREATE_INVESTIGATION_CARD, ActionType.UPDATE_SKILL,
    ActionType.WRITE_MEMORY, ActionType.GRADE_CONVERSATION,
    ActionType.CREATE_BUILD_CARD, ActionType.REMEASURE_METRIC,
    ActionType.ARCHIVE_STALE_PROPOSAL, ActionType.OPEN_PR,
})
_INFRA = frozenset({
    ActionType.ADJUST_CRON_SCHEDULE, ActionType.UNPAUSE_CRON,
    ActionType.CREATE_OODA_AGENT, ActionType.MODIFY_WEBHOOK_ROUTES,
})
_RELEASE = frozenset({
    ActionType.MERGE_STAGING, ActionType.ADJUST_THRESHOLD,
    ActionType.SEND_OPERATIONAL_MESSAGE, ActionType.CHANGE_CONVERSATION_FLOW,
    ActionType.CHANGE_BOOKING_RULE, ActionType.CHANGE_PATIENT_DATA_HANDLING,
    ActionType.EXTERNAL_PRODUCT_CHANGE, ActionType.DEPLOY_PRODUCTION,
})
_RESERVED_ACTIONS = frozenset({
    ActionType.BULK_SEND, ActionType.INCREASE_SPEND, ActionType.PIVOT_STRATEGY,
})
_SEND_ACTIONS = frozenset({
    ActionType.SEND_OPERATIONAL_MESSAGE, ActionType.CHANGE_CONVERSATION_FLOW,
    ActionType.CHANGE_BOOKING_RULE,
})


def _decision(outcome: DecisionOutcome, reason: str, owner: str = "chief_of_staff") -> Decision:
    return Decision(outcome, reason, owner)


def _ref(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def classify(action_type: ActionType, context: ActionContext) -> Decision:
    """Classify a proposed effect before execution; unknown inputs never authorize.

    Reserved effects are recognized even when an otherwise complete QA packet is
    supplied. An approval reference is not a verified, scope-matched decision.
    """
    if not isinstance(action_type, ActionType) or not isinstance(context, ActionContext):
        return _decision(DecisionOutcome.EVIDENCE_NEEDED, "unknown_action_or_context")
    if action_type in _RESERVED_ACTIONS:
        return _decision(DecisionOutcome.KEION_DECISION, "reserved_action", "keion")
    if (not isinstance(context.reserved_effects, frozenset)
            or any(not isinstance(effect, ReservedEffect) for effect in context.reserved_effects)):
        return _decision(DecisionOutcome.EVIDENCE_NEEDED, "invalid_or_unknown_reserved_effects")
    if context.reserved_effects:
        return _decision(DecisionOutcome.KEION_DECISION, "reserved_effect", "keion")
    if not all(_ref(value) for value in (context.evidence_ref, context.effect_scope, context.audience)):
        return _decision(DecisionOutcome.EVIDENCE_NEEDED, "missing_effect_evidence_or_audience")
    if context.same_purpose is not True:
        return _decision(DecisionOutcome.EVIDENCE_NEEDED, "purpose_unverified")
    if context.bounded_pilot is not True or context.bounded_safe_scope is not True:
        return _decision(DecisionOutcome.EVIDENCE_NEEDED, "scope_unverified")
    if context.dependency_ready is False:
        return _decision(DecisionOutcome.DEPENDENCY_WAIT, "dependency_not_ready")
    if context.dependency_ready is not True:
        return _decision(DecisionOutcome.EVIDENCE_NEEDED, "dependency_unverified")
    if action_type in _INFRA and context.opt_in is not True:
        return _decision(DecisionOutcome.EVIDENCE_NEEDED, "opt_in_unverified")
    if action_type in _INTERNAL:
        return _decision(DecisionOutcome.AGENT_AUTHORIZED, "bounded_internal_action", "agent")
    if action_type not in _INFRA | _RELEASE:
        return _decision(DecisionOutcome.EVIDENCE_NEEDED, "unmapped_action")
    if action_type in _SEND_ACTIONS and (type(context.send_cap) is not int or context.send_cap <= 0):
        return _decision(DecisionOutcome.EVIDENCE_NEEDED, "send_cap_unverified")
    if context.rollback_ready is not True:
        return _decision(DecisionOutcome.EVIDENCE_NEEDED, "rollback_unverified")
    if not all(_ref(value) for value in (
        context.qa_revision, context.project_ruling_ref, context.release_gate_ref,
    )):
        return _decision(DecisionOutcome.EVIDENCE_NEEDED, "release_packet_incomplete")
    if context.external_access_ready is not True:
        if context.external_access_ready is False:
            return _decision(DecisionOutcome.EXTERNAL_ACCESS_NEEDED, "external_access_missing", "release_engineer")
        return _decision(DecisionOutcome.EVIDENCE_NEEDED, "external_access_unverified")
    # The classifier does not perform QA, grant a project ruling, release, send,
    # or read production back. Even a completed packet cannot arm a live action.
    return _decision(DecisionOutcome.DARK_BUILD_ONLY, "await_ordered_release_and_witness")
