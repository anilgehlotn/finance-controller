"""Data models for the settlement-reconciliation tool.

GroundTruth is the eval-only answer key: it is used to score matcher/agent
output against reality and must never be passed to the matcher or the agent
itself.
"""

from enum import Enum

from pydantic import BaseModel, Field


class MatchType(str, Enum):
    EXACT = "EXACT"
    FUZZY_REF = "FUZZY_REF"
    DATE_SHIFTED = "DATE_SHIFTED"
    FEE_DEDUCTED = "FEE_DEDUCTED"
    COMBINED = "COMBINED"
    PARTIAL = "PARTIAL"
    FX_ROUNDING = "FX_ROUNDING"
    DUPLICATE_CANDIDATE = "DUPLICATE_CANDIDATE"
    ORPHAN_SETTLEMENT = "ORPHAN_SETTLEMENT"
    ORPHAN_TXN = "ORPHAN_TXN"


class ExceptionCode(str, Enum):
    AMOUNT_MISMATCH = "AMOUNT_MISMATCH"
    NO_COUNTERPARTY = "NO_COUNTERPARTY"
    DUPLICATE_CANDIDATE = "DUPLICATE_CANDIDATE"
    FX_ROUNDING = "FX_ROUNDING"
    PARTIAL_PAYMENT = "PARTIAL_PAYMENT"
    AMBIGUOUS_COMBINATION = "AMBIGUOUS_COMBINATION"


class Transaction(BaseModel):
    txn_id: str
    merchant_name: str
    gross_amount: float
    currency: str
    txn_date: str
    reference: str
    status: str = "captured"


class Settlement(BaseModel):
    settlement_id: str
    net_amount: float
    currency: str
    settlement_date: str
    bank_reference: str
    utr: str


class GroundTruth(BaseModel):
    settlement_id: str
    matched_txn_ids: list[str]
    match_type: MatchType
    fee_amount: float = 0
    tax_amount: float = 0
    notes: str = ""


class Resolution(BaseModel):
    settlement_id: str
    proposed_txn_ids: list[str]
    resolved: bool
    confidence: float = Field(ge=0, le=1)
    exception_code: ExceptionCode | None = None
    rationale: str = ""
