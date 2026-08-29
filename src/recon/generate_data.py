"""Synthetic data generator for settlement reconciliation.

Produces transactions.csv and settlements.csv -- the two inputs the matcher
and agent are allowed to see -- plus ground_truth.csv, a held-out answer key
used only for scoring. ground_truth.csv must never be read by the matcher or
the agent; it exists purely for evaluation.
"""

from __future__ import annotations

import argparse
import random
import string
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import polars as pl
from faker import Faker

from recon.schema import MatchType

# Tunable mix of case difficulty. Values must sum to 1.0.
DISTRIBUTION: dict[MatchType, float] = {
    MatchType.EXACT: 0.45,
    MatchType.FUZZY_REF: 0.12,
    MatchType.FEE_DEDUCTED: 0.12,
    MatchType.DATE_SHIFTED: 0.08,
    MatchType.COMBINED: 0.08,
    MatchType.PARTIAL: 0.04,
    MatchType.FX_ROUNDING: 0.04,
    MatchType.DUPLICATE_CANDIDATE: 0.03,
    MatchType.ORPHAN_SETTLEMENT: 0.02,
    MatchType.ORPHAN_TXN: 0.02,
}

# Cases with no true match; the matcher must flag these, not resolve them.
NON_MATCH_TYPES = {MatchType.ORPHAN_SETTLEMENT, MatchType.ORPHAN_TXN}

FEE_RATE = 0.02
GST_RATE = 0.18

# Fixed so output is identical run-to-run regardless of the calendar date.
ANCHOR_DATE = date(2026, 1, 1)

MERCHANT_SUFFIXES = [
    "Retail", "Traders", "Mart", "Foods", "Electronics", "Textiles",
    "Pharmacy", "Logistics", "Solutions", "Enterprises", "Stores",
    "Bazaar", "Hub", "Ventures", "Industries",
]

TXN_COLUMNS = [
    "txn_id", "merchant_name", "gross_amount", "currency", "txn_date",
    "reference", "status",
]
STL_COLUMNS = [
    "settlement_id", "net_amount", "currency", "settlement_date",
    "bank_reference", "utr",
]
GT_COLUMNS = [
    "settlement_id", "matched_txn_ids", "match_type", "fee_amount",
    "tax_amount", "notes",
]


def _round2(x: float) -> float:
    return round(x, 2)


def _fee_and_tax(gross: float) -> tuple[float, float]:
    fee = _round2(gross * FEE_RATE)
    tax = _round2(fee * GST_RATE)
    return fee, tax


def _build_merchant_pool(fake: Faker, n: int = 25) -> list[str]:
    names: set[str] = set()
    while len(names) < n:
        names.add(f"{fake.last_name()} {random.choice(MERCHANT_SUFFIXES)}")
    return sorted(names)


def _mangle_reference(reference: str) -> str:
    digits = reference.split("-", 1)[1]
    variant = random.choice(
        ["lower", "no_prefix", "extra_digit", "changed_digit", "txn_prefix"]
    )
    if variant == "lower":
        return reference.lower()
    if variant == "no_prefix":
        return digits
    if variant == "extra_digit":
        return f"{reference}{random.randint(0, 9)}"
    if variant == "changed_digit":
        pos = random.randrange(len(digits))
        new_digit = random.choice([d for d in string.digits if d != digits[pos]])
        return f"ORD-{digits[:pos]}{new_digit}{digits[pos + 1:]}"
    return f"TXN-{digits}"


def _random_date() -> date:
    return ANCHOR_DATE - timedelta(days=random.randint(1, 180))


def _random_gross() -> float:
    return _round2(random.uniform(150, 25000))


def _utr() -> str:
    return "UTR" + "".join(random.choices(string.digits, k=10))


@dataclass
class Counters:
    txn: int = 0
    stl: int = 0

    def next_txn_id(self) -> str:
        self.txn += 1
        return f"TXN{self.txn:06d}"

    def next_stl_id(self) -> str:
        self.stl += 1
        return f"STL{self.stl:06d}"


def _new_txn(
    counters: Counters,
    merchants: list[str],
    *,
    merchant: str | None = None,
    currency: str = "INR",
    status: str = "captured",
    txn_date: date | None = None,
    gross: float | None = None,
) -> tuple[dict, date]:
    txn_id = counters.next_txn_id()
    reference = txn_id.replace("TXN", "ORD-")
    if merchant is None:
        merchant = random.choice(merchants)
    if txn_date is None:
        txn_date = _random_date()
    if gross is None:
        gross = _random_gross()
    row = {
        "txn_id": txn_id,
        "merchant_name": merchant,
        "gross_amount": gross,
        "currency": currency,
        "txn_date": txn_date.isoformat(),
        "reference": reference,
        "status": status,
    }
    return row, txn_date


def _gt_row(
    settlement_id: str,
    txn_ids: list[str],
    match_type: MatchType,
    *,
    fee: float = 0.0,
    tax: float = 0.0,
    notes: str = "",
) -> dict:
    return {
        "settlement_id": settlement_id,
        "matched_txn_ids": ";".join(txn_ids),
        "match_type": match_type.value,
        "fee_amount": fee,
        "tax_amount": tax,
        "notes": notes,
    }


def _gen_exact(counters: Counters, merchants: list[str]) -> tuple[list[dict], dict | None, dict]:
    txn, txn_date = _new_txn(counters, merchants)
    settlement_id = counters.next_stl_id()
    stl_date = txn_date + timedelta(days=random.choice([0, 1, 2]))
    stl = {
        "settlement_id": settlement_id,
        "net_amount": txn["gross_amount"],
        "currency": txn["currency"],
        "settlement_date": stl_date.isoformat(),
        "bank_reference": txn["reference"],
        "utr": _utr(),
    }
    gt = _gt_row(settlement_id, [txn["txn_id"]], MatchType.EXACT)
    return [txn], stl, gt


def _gen_fuzzy_ref(counters: Counters, merchants: list[str]) -> tuple[list[dict], dict | None, dict]:
    txn, txn_date = _new_txn(counters, merchants)
    settlement_id = counters.next_stl_id()
    stl_date = txn_date + timedelta(days=random.choice([0, 1, 2]))
    stl = {
        "settlement_id": settlement_id,
        "net_amount": txn["gross_amount"],
        "currency": txn["currency"],
        "settlement_date": stl_date.isoformat(),
        "bank_reference": _mangle_reference(txn["reference"]),
        "utr": _utr(),
    }
    gt = _gt_row(
        settlement_id, [txn["txn_id"]], MatchType.FUZZY_REF,
        notes="bank_reference is a mangled variant of the transaction reference",
    )
    return [txn], stl, gt


def _gen_date_shifted(counters: Counters, merchants: list[str]) -> tuple[list[dict], dict | None, dict]:
    txn, txn_date = _new_txn(counters, merchants)
    settlement_id = counters.next_stl_id()
    stl_date = txn_date + timedelta(days=random.randint(3, 5))
    stl = {
        "settlement_id": settlement_id,
        "net_amount": txn["gross_amount"],
        "currency": txn["currency"],
        "settlement_date": stl_date.isoformat(),
        "bank_reference": txn["reference"],
        "utr": _utr(),
    }
    gt = _gt_row(
        settlement_id, [txn["txn_id"]], MatchType.DATE_SHIFTED,
        notes="settlement lands 3-5 days after the transaction, beyond normal T+1 lag",
    )
    return [txn], stl, gt


def _gen_fee_deducted(counters: Counters, merchants: list[str]) -> tuple[list[dict], dict | None, dict]:
    txn, txn_date = _new_txn(counters, merchants)
    fee, tax = _fee_and_tax(txn["gross_amount"])
    net = _round2(txn["gross_amount"] - fee - tax)
    settlement_id = counters.next_stl_id()
    stl_date = txn_date + timedelta(days=random.choice([0, 1, 2]))
    stl = {
        "settlement_id": settlement_id,
        "net_amount": net,
        "currency": txn["currency"],
        "settlement_date": stl_date.isoformat(),
        "bank_reference": txn["reference"],
        "utr": _utr(),
    }
    gt = _gt_row(
        settlement_id, [txn["txn_id"]], MatchType.FEE_DEDUCTED, fee=fee, tax=tax,
        notes="net = gross - platform fee - GST on fee",
    )
    return [txn], stl, gt


def _gen_combined(counters: Counters, merchants: list[str]) -> tuple[list[dict], dict | None, dict]:
    merchant = random.choice(merchants)
    base_date = _random_date()
    n = random.randint(2, 5)
    txns: list[dict] = []
    dates: list[date] = []
    total_net = 0.0
    fee_sum = 0.0
    tax_sum = 0.0
    for _ in range(n):
        candidate_date = base_date + timedelta(days=random.randint(0, 1))
        txn, txn_date = _new_txn(counters, merchants, merchant=merchant, txn_date=candidate_date)
        fee, tax = _fee_and_tax(txn["gross_amount"])
        total_net += txn["gross_amount"] - fee - tax
        fee_sum += fee
        tax_sum += tax
        txns.append(txn)
        dates.append(txn_date)
    settlement_id = counters.next_stl_id()
    stl_date = max(dates) + timedelta(days=random.choice([0, 1, 2]))
    stl = {
        "settlement_id": settlement_id,
        "net_amount": _round2(total_net),
        "currency": "INR",
        "settlement_date": stl_date.isoformat(),
        "bank_reference": f"BATCH-{counters.stl:04d}",
        "utr": _utr(),
    }
    gt = _gt_row(
        settlement_id, [t["txn_id"] for t in txns], MatchType.COMBINED,
        fee=_round2(fee_sum), tax=_round2(tax_sum),
        notes=f"{n} same-merchant transactions rolled into one settlement",
    )
    return txns, stl, gt


def _gen_partial(counters: Counters, merchants: list[str]) -> tuple[list[dict], dict | None, dict]:
    txn, txn_date = _new_txn(counters, merchants)
    frac = random.uniform(0.4, 0.7)
    net = _round2(txn["gross_amount"] * frac)
    settlement_id = counters.next_stl_id()
    stl_date = txn_date + timedelta(days=random.choice([0, 1, 2]))
    stl = {
        "settlement_id": settlement_id,
        "net_amount": net,
        "currency": txn["currency"],
        "settlement_date": stl_date.isoformat(),
        "bank_reference": txn["reference"],
        "utr": _utr(),
    }
    gt = _gt_row(
        settlement_id, [txn["txn_id"]], MatchType.PARTIAL,
        notes=f"partial payment: settlement is {frac:.0%} of gross",
    )
    return [txn], stl, gt


def _gen_fx_rounding(counters: Counters, merchants: list[str]) -> tuple[list[dict], dict | None, dict]:
    gross = _round2(random.uniform(10, 900))
    txn, txn_date = _new_txn(counters, merchants, currency="USD", gross=gross)
    rate = random.uniform(82.5, 84.5)
    drift = random.uniform(-0.4, 0.4)
    net = _round2(gross * rate + drift)
    settlement_id = counters.next_stl_id()
    stl_date = txn_date + timedelta(days=random.choice([0, 1, 2]))
    stl = {
        "settlement_id": settlement_id,
        "net_amount": net,
        "currency": "INR",
        "settlement_date": stl_date.isoformat(),
        "bank_reference": txn["reference"],
        "utr": _utr(),
    }
    gt = _gt_row(
        settlement_id, [txn["txn_id"]], MatchType.FX_ROUNDING,
        notes=f"USD->INR at ~{rate:.4f} with small rounding drift",
    )
    return [txn], stl, gt


def _gen_duplicate_candidate(
    counters: Counters, merchants: list[str]
) -> tuple[list[dict], dict | None, dict]:
    merchant = random.choice(merchants)
    txn_date = _random_date()
    gross = _random_gross()
    txn_a, _ = _new_txn(counters, merchants, merchant=merchant, txn_date=txn_date, gross=gross)
    txn_b, _ = _new_txn(counters, merchants, merchant=merchant, txn_date=txn_date, gross=gross)
    true_txn = random.choice([txn_a, txn_b])
    settlement_id = counters.next_stl_id()
    stl_date = txn_date + timedelta(days=random.choice([0, 1, 2]))
    stl = {
        "settlement_id": settlement_id,
        "net_amount": gross,
        "currency": txn_a["currency"],
        "settlement_date": stl_date.isoformat(),
        "bank_reference": f"PAY-{counters.stl:04d}",
        "utr": _utr(),
    }
    gt = _gt_row(
        settlement_id, [true_txn["txn_id"]], MatchType.DUPLICATE_CANDIDATE,
        notes="two identical-amount/date/merchant transactions exist; only one truly settled",
    )
    return [txn_a, txn_b], stl, gt


def _gen_orphan_settlement(
    counters: Counters, merchants: list[str]
) -> tuple[list[dict], dict | None, dict]:
    settlement_id = counters.next_stl_id()
    stl = {
        "settlement_id": settlement_id,
        "net_amount": _random_gross(),
        "currency": "INR",
        "settlement_date": _random_date().isoformat(),
        "bank_reference": f"REV-{counters.stl:04d}",
        "utr": _utr(),
    }
    gt = _gt_row(
        settlement_id, [], MatchType.ORPHAN_SETTLEMENT,
        notes="no matching transaction exists",
    )
    return [], stl, gt


def _gen_orphan_txn(counters: Counters, merchants: list[str]) -> tuple[list[dict], dict | None, dict]:
    status = random.choice(["failed", "pending"])
    txn, _ = _new_txn(counters, merchants, status=status)
    settlement_id = f"NOSTL-{txn['txn_id']}"
    gt = _gt_row(
        settlement_id, [txn["txn_id"]], MatchType.ORPHAN_TXN,
        notes=f"transaction never settled (status={status})",
    )
    return [txn], None, gt


_GENERATORS = {
    MatchType.EXACT: _gen_exact,
    MatchType.FUZZY_REF: _gen_fuzzy_ref,
    MatchType.DATE_SHIFTED: _gen_date_shifted,
    MatchType.FEE_DEDUCTED: _gen_fee_deducted,
    MatchType.COMBINED: _gen_combined,
    MatchType.PARTIAL: _gen_partial,
    MatchType.FX_ROUNDING: _gen_fx_rounding,
    MatchType.DUPLICATE_CANDIDATE: _gen_duplicate_candidate,
    MatchType.ORPHAN_SETTLEMENT: _gen_orphan_settlement,
    MatchType.ORPHAN_TXN: _gen_orphan_txn,
}


def _allocate_case_types(count: int) -> list[MatchType]:
    types = list(DISTRIBUTION.keys())
    weights = list(DISTRIBUTION.values())
    raw_counts = [w * count for w in weights]
    counts = [int(x) for x in raw_counts]
    remainder = count - sum(counts)
    order = sorted(range(len(types)), key=lambda i: raw_counts[i] - counts[i], reverse=True)
    for i in order[:remainder]:
        counts[i] += 1
    case_types: list[MatchType] = []
    for match_type, n in zip(types, counts):
        case_types.extend([match_type] * n)
    random.shuffle(case_types)
    return case_types


def build_dataset(seed: int, count: int) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Build the three datasets in memory. Reseeds internally, so calling this
    twice with the same (seed, count) always yields identical output."""
    random.seed(seed)
    Faker.seed(seed)
    fake = Faker("en_IN")
    merchants = _build_merchant_pool(fake)
    counters = Counters()

    txn_rows: list[dict] = []
    stl_rows: list[dict] = []
    gt_rows: list[dict] = []
    for match_type in _allocate_case_types(count):
        txns, stl, gt = _GENERATORS[match_type](counters, merchants)
        txn_rows.extend(txns)
        if stl is not None:
            stl_rows.append(stl)
        gt_rows.append(gt)

    random.shuffle(txn_rows)
    random.shuffle(stl_rows)

    txn_df = pl.DataFrame(txn_rows).select(TXN_COLUMNS)
    stl_df = pl.DataFrame(stl_rows).select(STL_COLUMNS)
    gt_df = pl.DataFrame(gt_rows).select(GT_COLUMNS)
    return txn_df, stl_df, gt_df


def _print_summary(txn_df: pl.DataFrame, stl_df: pl.DataFrame, gt_df: pl.DataFrame, count: int) -> None:
    print(f"transactions: {txn_df.height}")
    print(f"settlements:  {stl_df.height}")
    print("match type distribution:")
    counted = gt_df.group_by("match_type").len()
    count_map = dict(zip(counted["match_type"], counted["len"]))
    for match_type in DISTRIBUTION:
        n = count_map.get(match_type.value, 0)
        pct = 100 * n / count if count else 0.0
        print(f"  {match_type.value:<22} {n:>5}  ({pct:5.1f}%)")
    real_match = sum(
        count_map.get(mt.value, 0) for mt in DISTRIBUTION if mt not in NON_MATCH_TYPES
    )
    print(f"{real_match}/{count} cases have a real match; the rest must be flagged, not matched.")


def generate(seed: int, count: int, output_dir: Path) -> None:
    txn_df, stl_df, gt_df = build_dataset(seed, count)
    output_dir.mkdir(parents=True, exist_ok=True)
    txn_df.write_csv(output_dir / "transactions.csv")
    stl_df.write_csv(output_dir / "settlements.csv")
    gt_df.write_csv(output_dir / "ground_truth.csv")
    _print_summary(txn_df, stl_df, gt_df, count)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic settlement reconciliation data.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--count", type=int, default=600)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    args = parser.parse_args(argv)
    generate(seed=args.seed, count=args.count, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
