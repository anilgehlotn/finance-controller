import polars as pl

from recon.generate_data import build_dataset
from recon.schema import MatchType


def test_same_seed_is_reproducible():
    _, _, gt1 = build_dataset(seed=7, count=200)
    _, _, gt2 = build_dataset(seed=7, count=200)
    assert gt1.equals(gt2)


def test_fee_deducted_math_is_exact():
    txn_df, stl_df, gt_df = build_dataset(seed=7, count=400)
    fee_rows = gt_df.filter(pl.col("match_type") == MatchType.FEE_DEDUCTED.value)
    assert fee_rows.height > 0

    joined = fee_rows.join(stl_df, on="settlement_id").join(
        txn_df, left_on="matched_txn_ids", right_on="txn_id"
    )
    assert joined.height == fee_rows.height

    for row in joined.iter_rows(named=True):
        expected_net = row["gross_amount"] - row["fee_amount"] - row["tax_amount"]
        assert abs(row["net_amount"] - expected_net) < 0.01


def test_orphan_settlement_has_no_matched_txns():
    _, _, gt_df = build_dataset(seed=7, count=300)
    orphan_rows = gt_df.filter(pl.col("match_type") == MatchType.ORPHAN_SETTLEMENT.value)
    assert orphan_rows.height > 0
    assert (orphan_rows["matched_txn_ids"] == "").all()
