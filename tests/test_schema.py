import pytest
from pydantic import ValidationError

from recon.schema import MatchType, Resolution


def test_match_type_exact_value():
    assert MatchType.EXACT.value == "EXACT"


def test_resolution_confidence_out_of_range_raises():
    with pytest.raises(ValidationError):
        Resolution(
            settlement_id="s1",
            proposed_txn_ids=["t1"],
            resolved=True,
            confidence=1.2,
        )
