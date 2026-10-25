import numpy as np

from finmlkit.bar.data_model import TradesData


def test_side_stays_with_its_trade_when_input_is_not_in_id_order():
    """Binance dumps are not always in id order; sorting must carry is_buyer_maker along with price/time/amount."""
    n = 1000
    ids = np.arange(1, n + 1, dtype=np.int64)
    ts = (1_700_000_000_000 + ids * 5).astype(np.int64)            # ms, one trade per timestamp: nothing merges
    px = 100.0 + ids * 0.01
    qty = np.ones(n)
    buyer_maker = (ids % 3 == 0)                                    # a side that is a function of the trade
    order = np.random.default_rng(0).permutation(n)                 # shuffled, as in an out-of-order dump
    td = TradesData(ts[order], px[order], qty[order], id=ids[order], is_buyer_maker=buyer_maker[order],
                    timestamp_unit="ms", preprocess=True)
    out = td.data
    trade_id = np.rint((out["price"].to_numpy() - 100.0) / 0.01).astype(np.int64)
    expected = np.where(trade_id % 3 == 0, -1, 1)                   # buyer maker => seller aggressor => -1
    assert (out["side"].to_numpy() == expected).all()


def test_sorted_input_is_unchanged():
    n = 50
    ids = np.arange(1, n + 1, dtype=np.int64)
    ts = (1_700_000_000_000 + ids * 5).astype(np.int64)
    td = TradesData(ts, 100.0 + ids, np.ones(n), id=ids, is_buyer_maker=(ids % 2 == 0),
                    timestamp_unit="ms", preprocess=True)
    assert (td.data["side"].to_numpy() == np.where(ids % 2 == 0, -1, 1)).all()
